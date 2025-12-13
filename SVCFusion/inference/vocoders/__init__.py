from dataclasses import dataclass
import os
from loguru import logger
import torch
from torchaudio.transforms import Resample

from SVCFusion.i18n import I
from utils import Dummy; gr = Dummy()

from .nsf_hifigan.nvSTFT import STFT
from .nsf_hifigan.models import load_model, load_config


class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if vocoder_type == "nsf-hifigan":
            self.vocoder = NsfHifiGAN(vocoder_ckpt, device=device)
        elif vocoder_type == "nsf-hifigan-log10":
            self.vocoder = NsfHifiGANLog10(vocoder_ckpt, device=device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")

        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()

    def extract(self, audio, sample_rate=0, keyshift=0):
        # resample
        if sample_rate == self.vocoder_sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate, self.vocoder_sample_rate, lowpass_filter_width=128
                ).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)

        # extract
        mel = self.vocoder.extract(audio_res, keyshift=keyshift)  # B, n_frames, bins
        return mel

    def infer(self, mel, f0):
        f0 = f0[:, : mel.size(1), 0]  # B, n_frames
        audio = self.vocoder(mel, f0)
        return audio


class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)
        self.stft = STFT(
            self.h.sampling_rate,
            self.h.num_mels,
            self.h.n_fft,
            self.h.win_size,
            self.h.hop_size,
            self.h.fmin,
            self.h.fmax,
        )

    def sample_rate(self):
        return self.h.sampling_rate

    def hop_size(self):
        return self.h.hop_size

    def dimension(self):
        return self.h.num_mels

    def extract(self, audio, keyshift=0):
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(
            1, 2
        )  # B, n_frames, bins
        return mel

    def forward(self, mel, f0):
        if self.model is None:
            print("| Load HifiGAN: ", self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class NsfHifiGANLog10(NsfHifiGAN):
    def forward(self, mel, f0):
        if self.model is None:
            print("| Load HifiGAN: ", self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


# 下面都不是从原项目摘抄的


@dataclass
class VocoderInfo:
    name: str
    ckpt: str
    type: str


shared_vocoder: NsfHifiGAN = None
shared_vocoder_info: VocoderInfo = None

vocoders: dict[str, VocoderInfo] = {
    "Kouon NSF HifiGAN 1031": VocoderInfo(
        name="Kouon NSF HifiGAN 1031",
        ckpt="pretrain/vocoder/kouon_nsf-hifigan_1031_generators/kouon_nsf-hifigan_1031_44100hz_512hop_128bin.ckpt",
        type="nsf-hifigan",
    ),
    "Kouon PC NSF HifiGAN 1029": VocoderInfo(
        name="Kouon PC NSF HifiGAN 1029",
        ckpt="pretrain/vocoder/kouon_pc_nsf-hifigan_1029_generators/kouon_pc_nsf-hifigan_1029_44100hz_512hop_128bin.ckpt",
        type="nsf-hifigan",
    ),
    # "OpenVPI NSF HifiGAN 20221211": VocoderInfo(
    #     name="OpenVPI NSF HifiGAN 20221211",
    #     ckpt="pretrain/nsf_hifigan/model",
    #     type="nsf-hifigan",
    # ),
    "OpenVPI PC NSF HifiGAN 2025.02": VocoderInfo(
        name="PC NSF HifiGAN 2025.02",
        ckpt="pretrain/vocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt",
        type="nsf-hifigan",
    ),
}


def get_vocoder_keys():
    result = []
    for name, info in vocoders.items():
        if not os.path.exists(info.ckpt):
            pass
            # logger.warning(
            #     f"Vocoder ckpt not found: {info.ckpt}, removing '{name}' from vocoders."
            # )
        else:
            result.append(name)
    return result


def set_shared_vocoder(
    vocoder_name,
    device,
):
    global shared_vocoder, shared_vocoder_info
    if vocoder_name not in vocoders:
        raise gr.Error(I.common_infer.unknown_vocoder_tip)

    vocoder_info = vocoders[vocoder_name]
    vocoder_type = vocoder_info.type
    vocoder_ckpt = vocoder_info.ckpt
    logger.info(f"Loading vocoder: {vocoder_type} {vocoder_ckpt} device={device}")
    shared_vocoder = Vocoder(vocoder_type, vocoder_ckpt, device=device)
    shared_vocoder_info = vocoder_info


def get_shared_vocoder():
    if shared_vocoder is None:
        raise gr.Error(I.common_infer.vocoder_not_loaded_tip)
    return shared_vocoder


def get_shared_vocoder_info():
    if shared_vocoder_info is None:
        raise gr.Error(I.common_infer.vocoder_not_loaded_tip)
    return shared_vocoder_info
