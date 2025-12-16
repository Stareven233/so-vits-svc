from util import Dummy; gr = Dummy()
import librosa
from typing import Optional, Any

from loguru import logger

from SVCFusion.file import hash_file
from SVCFusion.i18n import I
from SVCFusion.inference.vocoders import (
    get_shared_vocoder,
    get_vocoder_keys,
    set_shared_vocoder,
)
from util.sov_utils import get_f0_predictor

import soundfile as sf
import torch


class VocoderKeyShift:
    def __init__(self) -> None:
        self.vocoder_dropdown = gr.Dropdown(
            value=self.update_vocoder,
            label=I.vocoder_key_shift.vocoder_dropdown_label,
            info=I.vocoder_key_shift.update_vocoder_info,
        )
        self.keyshift_slider = gr.Slider(
            minimum=-12,
            maximum=12,
            step=1,
            value=0,
            label=I.vocoder_key_shift.keyshift_slider_label,
            info=I.vocoder_key_shift.keyshift_slider_info,
        )
        self.audio_input = gr.Audio(
            label=I.vocoder_key_shift.audio_input_label,
            type="filepath",
            interactive=True,
        )
        self.submit_button = gr.Button(
            value=I.vocoder_key_shift.submit_btn_value,
            variant="primary",
        )
        self.output_audio = gr.Audio(
            label=I.vocoder_key_shift.output_audio_label,
            type="filepath",
            interactive=False,
        )
        self.submit_button.click(
            self.process,
            inputs=[self.vocoder_dropdown, self.keyshift_slider, self.audio_input],
            outputs=self.output_audio,
        )

    def update_vocoder(self) -> Any:
        vocoder_keys = get_vocoder_keys()
        return gr.update(
            value=vocoder_keys[0],
            choices=vocoder_keys,
        )

    def process(
        self,
        vocoder_name: Optional[str],
        key_shift: int,
        audio_path: Optional[str],
    ) -> Optional[str]:
        if vocoder_name is None or audio_path is None:
            return None
        if vocoder_name not in get_vocoder_keys():
            return None
        set_shared_vocoder(vocoder_name, "cpu")
        vocoder = get_shared_vocoder()
        # 用librosa加载音频
        audio_data, sample_rate = librosa.load(audio_path, sr=44100)
        if audio_data.ndim > 1:
            audio_data = librosa.to_mono(audio_data)
        # 提取F0
        f0_predictor = get_f0_predictor(
            "rmvpe",
            hop_length=512,
            sampling_rate=44100,
            device="cpu",
            threshold=0.1,
        )
        f0 = f0_predictor.compute_f0(audio_data)
        # 升降调
        f0_shifted = f0 * 2 ** (key_shift / 12.0)
        # 提取mel
        audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
        mel = vocoder.extract(audio_tensor, 44100)
        mel = mel.squeeze().float().unsqueeze(0)
        # 合成音频
        wav = vocoder.infer(
            mel, torch.from_numpy(f0_shifted).float().unsqueeze(0).unsqueeze(-1)
        )
        output_path = f"tmp/vocoder_keyshift/{hash_file(audio_path)}.wav"
        sf.write(output_path, wav.squeeze().cpu().numpy(), 44100)
        logger.info(
            I.vocoder_key_shift.process_success_log.format(output_path=output_path)
        )
        return output_path
