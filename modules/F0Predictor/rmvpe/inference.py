import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .constants import N_MELS, SAMPLE_RATE, WINDOW_LENGTH, MEL_FMIN, MEL_FMAX
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_cents, to_viterbi_cents


class RMVPE:
    def __init__(self, model_path, device=None, dtype=torch.float32, hop_length=160):
        self.resample_kernel = {}
        # 添加轻量级缓存，仅缓存计算结果不影响模型结构
        self._cache_enabled = True
        self._result_cache = {}
        self._cache_size_limit = 5

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        model = E2E0(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location=torch.device(self.device))
        model.load_state_dict(ckpt["model"])
        model = model.to(dtype).to(self.device)
        model.eval()

        # 设置为推理模式，可以提升性能
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.dtype = dtype
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        )
        self.resample_kernel = {}

    def _get_audio_hash(self, audio_tensor):
        """为音频生成简单的哈希作为缓存键"""
        if not self._cache_enabled:
            return None
        try:
            # 使用音频的统计特征生成哈希
            return hash(
                (
                    audio_tensor.shape,
                    float(audio_tensor.mean()),
                    float(audio_tensor.std()),
                )
            )
        except:
            return None

    def _clear_cache_if_needed(self):
        """清理缓存避免内存泄漏"""
        if len(self._result_cache) > self._cache_size_limit:
            # 清除一半的缓存
            keys_to_remove = list(self._result_cache.keys())[
                : -self._cache_size_limit // 2
            ]
            for key in keys_to_remove:
                del self._result_cache[key]

    def mel2hidden(self, mel):
        # 使用无梯度上下文优化推理
        with torch.no_grad():
            n_frames = mel.shape[-1]
            # 优化padding计算
            remainder = n_frames % 32
            if remainder != 0:
                pad_size = 32 - remainder
                mel = F.pad(mel, (0, pad_size), mode="constant")

            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        # 使用向量化操作替代循环
        with torch.no_grad():
            if use_viterbi:
                cents_pred = to_viterbi_cents(hidden, thred=thred)
            else:
                cents_pred = to_local_average_cents(hidden, thred=thred)

            # 向量化F0计算
            if isinstance(cents_pred, torch.Tensor):
                # 预分配结果张量
                f0 = torch.zeros_like(cents_pred, dtype=self.dtype, device=self.device)
                # 使用掩码避免循环
                nonzero_mask = cents_pred != 0
                if nonzero_mask.any():
                    f0[nonzero_mask] = 10 * (2 ** (cents_pred[nonzero_mask] / 1200))
            else:
                # numpy数组转换
                cents_tensor = torch.tensor(
                    cents_pred, dtype=self.dtype, device=self.device
                )
                f0 = torch.zeros_like(cents_tensor)
                nonzero_mask = cents_tensor != 0
                if nonzero_mask.any():
                    f0[nonzero_mask] = 10 * (2 ** (cents_tensor[nonzero_mask] / 1200))

        return f0

    def infer_from_audio(self, audio, sample_rate=16000, thred=0.05, use_viterbi=False):
        # 检查缓存
        audio_hash = self._get_audio_hash(audio)
        cache_key = (
            (audio_hash, sample_rate, thred, use_viterbi) if audio_hash else None
        )

        if cache_key and cache_key in self._result_cache:
            return self._result_cache[cache_key].clone()

        # 确保输入格式正确
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.dtype).to(self.device)

        # 优化重采样逻辑
        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = (
                    Resample(sample_rate, 16000, lowpass_filter_width=128)
                    .to(self.dtype)
                    .to(self.device)
                )
            audio_res = self.resample_kernel[key_str](audio)

        # 确保mel_extractor在正确设备上（一次性检查）
        if not hasattr(self.mel_extractor, "_device_set"):
            self.mel_extractor = self.mel_extractor.to(self.device)
            self.mel_extractor._device_set = True

        # 计算频谱和推理
        mel = self.mel_extractor(audio_res, center=True).to(self.dtype)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden.squeeze(0), thred=thred, use_viterbi=use_viterbi)

        # 缓存结果
        if cache_key:
            self._result_cache[cache_key] = f0.clone()
            self._clear_cache_if_needed()

        return f0
