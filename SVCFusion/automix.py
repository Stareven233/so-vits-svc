"""
Developed by: C_Zim(Chai🍊)
Date: 25/2/26
Version: 0.3
Description:
    此项目仅供娱乐，不得用于商业用途
    这只是个模板，请根据自己的需求进行修改
    请确保你有一定的混音知识和经验，否则只会越改越差
"""

import tempfile
import warnings
from typing import Tuple, List
from enum import Enum

import librosa
import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard,
    Mix,
    Gain,
    HighpassFilter,
    PeakFilter,
    HighShelfFilter,
    LowShelfFilter,
    Delay,
    Invert,
    Compressor,
    Reverb,
    Limiter,
)
from pedalboard.io import AudioFile

warnings.filterwarnings("ignore")


class ReverbLevel(Enum):
    """混响等级枚举"""

    DRY = 0  # 干声，无混响
    SUBTLE = 1  # 微妙，轻微混响
    LIGHT = 2  # 轻度混响
    MODERATE = 3  # 中度混响（默认）
    HEAVY = 4  # 重度混响
    EXTREME = 5  # 极重混响


class MusicGenre(Enum):
    """音乐风格枚举"""

    POP = "pop"  # 流行音乐
    ROCK = "rock"  # 摇滚音乐
    JAZZ = "jazz"  # 爵士音乐
    ELECTRONIC = "electronic"  # 电子音乐
    FOLK = "folk"  # 民谣/原声
    CLASSICAL = "classical"  # 古典音乐


class VoiceType(Enum):
    """人声类型枚举"""

    MALE_LOW = "male_low"  # 男声低音
    MALE_HIGH = "male_high"  # 男声高音
    FEMALE = "female"  # 女声
    RAP = "rap"  # 说唱
    VOCAL = "vocal"  # 美声/古典


class DeEsserStrength(Enum):
    """去齿音强度枚举"""

    OFF = 0  # 关闭
    LIGHT = 1  # 轻微
    MODERATE = 2  # 中等
    HEAVY = 3  # 强烈


class CompressionStrength(Enum):
    """压缩强度枚举"""

    LIGHT = 1  # 轻压缩
    MODERATE = 2  # 标准压缩
    HEAVY = 3  # 重压缩


class EQStyle(Enum):
    """EQ风格枚举"""

    NEUTRAL = "neutral"  # 中性
    BRIGHT = "bright"  # 明亮
    WARM = "warm"  # 温暖
    VINTAGE = "vintage"  # 复古


class EchoLevel(Enum):
    """回声等级枚举"""

    OFF = 0  # 关闭
    SUBTLE = 1  # 微妙回声
    LIGHT = 2  # 轻度回声
    MODERATE = 3  # 中度回声
    HEAVY = 4  # 重度回声
    EXTREME = 5  # 极重回声


class TimeCalculator:
    """计算基于音频节拍的时间参数"""

    def __init__(self, inst_path: str):
        y, sr = librosa.load(inst_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(int(tempo), 0)
        if bpm >= 100:
            bpm = bpm / 2
        self.basic_time = 60000 / bpm
        self.times = {
            "pre_delay": self.reverb_pre_delay(),
            "release": self.compressor_release(),
        }

    def _calculate_time(self, times: List[float]) -> List[float]:
        """计算时间分割"""
        stop = 0
        for time in times[:]:  # 使用副本避免修改原列表
            half_time = time / 2
            times.append(half_time)
            stop += 1
            if stop >= 15:
                break
        return times[:]

    def _select_time(
        self,
        time_lists: List[float],
        standard_value: float,
        standard_range: float,
        double_mode: bool = False,
    ) -> float:
        """根据标准值选择最合适的时间"""
        if min(time_lists) >= standard_range:
            return standard_value * 2 if double_mode else standard_value

        min_diff = float("inf")
        closest_num = standard_value

        for time_value in time_lists:
            diff = abs(time_value - standard_value)
            if diff < min_diff:
                min_diff = diff
                closest_num = time_value

        return closest_num * 2 if double_mode else closest_num

    def _note(self, rate: float, mode: int) -> List[float]:
        """计算音符时间"""
        if mode == 0:
            note = self.basic_time * rate
        elif mode == 1:
            note = self.basic_time / rate
        else:
            raise ValueError("Mode must be 0 or 1")

        dot = note * 1.5
        triplet = note * 2 / 3
        bases = [note, dot, triplet]
        fulls = self._calculate_time(bases)

        return sorted(fulls)

    def reverb_pre_delay(self) -> Tuple[float, float, float, float]:
        """计算混响预延迟时间"""
        pre_delay_raws = self._note(8, 1)
        pre_delays = [round(pre_delay, 2) for pre_delay in pre_delay_raws]

        room_er = self._select_time(pre_delays, 0.6, 1, True)
        room_lr = self._select_time(pre_delays, 2, 4, True)
        plate = self._select_time(pre_delays, 10, 20, True)
        hall = self._select_time(pre_delays, 20, 40, True)

        return room_er, room_lr, plate, hall

    def compressor_release(self) -> Tuple[float, float, float, float]:
        """计算压缩器释放时间"""
        release_raws = self._note(2, 0)
        releases = [round(release, 1) for release in release_raws]

        fast = self._select_time(releases, 100, 200)
        medium = self._select_time(releases, 350, 500)
        slow = self._select_time(releases, 500, 1000)
        limiter = self._select_time(releases, 450, 800)

        return fast, medium, slow, limiter


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    """加载音频文件"""
    with AudioFile(path).resampled_to(sample_rate) as audio:
        data = audio.read(audio.frames)
    return data


def get_genre_parameters(genre: MusicGenre) -> dict:
    """根据音乐风格获取参数"""
    parameters = {
        MusicGenre.POP: {
            "vocal_brightness": 0,
            "compression_ratio": 2.5,
            "reverb_adjustment": 0,
            "high_shelf_gain": 0,
        },
        MusicGenre.ROCK: {
            "vocal_brightness": 2,
            "compression_ratio": 3.5,
            "reverb_adjustment": -3,
            "high_shelf_gain": 2,
        },
        MusicGenre.JAZZ: {
            "vocal_brightness": -1,
            "compression_ratio": 1.8,
            "reverb_adjustment": 2,
            "high_shelf_gain": -1,
        },
        MusicGenre.ELECTRONIC: {
            "vocal_brightness": 3,
            "compression_ratio": 4.0,
            "reverb_adjustment": 0,
            "high_shelf_gain": 3,
        },
        MusicGenre.FOLK: {
            "vocal_brightness": -2,
            "compression_ratio": 1.5,
            "reverb_adjustment": 1,
            "high_shelf_gain": -2,
        },
        MusicGenre.CLASSICAL: {
            "vocal_brightness": -1,
            "compression_ratio": 1.2,
            "reverb_adjustment": 4,
            "high_shelf_gain": 0,
        },
    }
    return parameters[genre]


def get_voice_type_parameters(voice_type: VoiceType) -> dict:
    """根据人声类型获取参数"""
    parameters = {
        VoiceType.MALE_LOW: {
            "highpass_freq": 80,
            "presence_freq": 2500,
            "presence_gain": 2,
            "brightness_freq": 8000,
            "brightness_gain": 1,
        },
        VoiceType.MALE_HIGH: {
            "highpass_freq": 100,
            "presence_freq": 3000,
            "presence_gain": 3,
            "brightness_freq": 10000,
            "brightness_gain": 2,
        },
        VoiceType.FEMALE: {
            "highpass_freq": 120,
            "presence_freq": 3500,
            "presence_gain": 2.5,
            "brightness_freq": 12000,
            "brightness_gain": 2.5,
        },
        VoiceType.RAP: {
            "highpass_freq": 150,
            "presence_freq": 4000,
            "presence_gain": 4,
            "brightness_freq": 8000,
            "brightness_gain": 3,
        },
        VoiceType.VOCAL: {
            "highpass_freq": 60,
            "presence_freq": 2000,
            "presence_gain": 1,
            "brightness_freq": 10000,
            "brightness_gain": 1,
        },
    }
    return parameters[voice_type]


def get_deesser_parameters(strength: DeEsserStrength) -> dict:
    """根据去齿音强度获取参数"""
    parameters = {
        DeEsserStrength.OFF: {"freq": 0, "gain": 0, "q": 1},
        DeEsserStrength.LIGHT: {"freq": 6500, "gain": -2, "q": 2},
        DeEsserStrength.MODERATE: {"freq": 6000, "gain": -4, "q": 2.5},
        DeEsserStrength.HEAVY: {"freq": 5500, "gain": -6, "q": 3},
    }
    return parameters[strength]


def get_eq_style_parameters(style: EQStyle) -> dict:
    """根据EQ风格获取参数"""
    parameters = {
        EQStyle.NEUTRAL: {
            "low_shelf_gain": 0,
            "mid_gain": 0,
            "high_shelf_gain": 0,
        },
        EQStyle.BRIGHT: {
            "low_shelf_gain": -1,
            "mid_gain": 1,
            "high_shelf_gain": 3,
        },
        EQStyle.WARM: {
            "low_shelf_gain": 2,
            "mid_gain": -1,
            "high_shelf_gain": -2,
        },
        EQStyle.VINTAGE: {
            "low_shelf_gain": 1,
            "mid_gain": 2,
            "high_shelf_gain": -3,
        },
    }
    return parameters[style]


def get_echo_parameters(level: EchoLevel, basic_time: float) -> dict:
    """根据回声等级获取回声参数

    Args:
        level: 回声等级
        basic_time: 基础时间（从TimeCalculator获取）

    Returns:
        回声参数字典
    """
    # 计算更短的延迟时间，以固定值为主，节拍为辅
    parameters = {
        EchoLevel.OFF: {
            "delay_time": 0,
            "feedback": 0,
            "wet_level": 0,
            "gain_adjustment": 0,
        },
        EchoLevel.SUBTLE: {
            "delay_time": 60,  # 固定60毫秒，非常微妙
            "feedback": 0.05,  # 极低反馈
            "wet_level": 0.08,  # 很低的湿声比例
            "gain_adjustment": -4,  # 降低增益避免过强
        },
        EchoLevel.LIGHT: {
            "delay_time": 120,  # 固定120毫秒
            "feedback": 0.1,  # 低反馈
            "wet_level": 0.15,  # 适中的湿声比例
            "gain_adjustment": -2,
        },
        EchoLevel.MODERATE: {
            "delay_time": min(basic_time / 8, 200),  # 最多200毫秒
            "feedback": 0.2,
            "wet_level": 0.25,
            "gain_adjustment": 0,
        },
        EchoLevel.HEAVY: {
            "delay_time": min(basic_time / 4, 350),  # 最多350毫秒
            "feedback": 0.3,
            "wet_level": 0.4,
            "gain_adjustment": 1,
        },
        EchoLevel.EXTREME: {
            "delay_time": min(basic_time / 2, 500),  # 最多500毫秒
            "feedback": 0.45,
            "wet_level": 0.6,
            "gain_adjustment": 2,
        },
    }
    return parameters[level]


def create_echo_chain(
    echo_level: EchoLevel = EchoLevel.OFF,
    basic_time: float = 500,
) -> Pedalboard:
    """创建回声处理链"""
    echo_params = get_echo_parameters(echo_level, basic_time)

    # 如果回声关闭，返回空的处理链
    if echo_level == EchoLevel.OFF:
        return Pedalboard([Gain(0)])

    # 将毫秒转换为秒
    delay_time_seconds = echo_params["delay_time"] / 1000

    return Pedalboard(
        [
            Delay(
                delay_time_seconds, echo_params["feedback"], echo_params["wet_level"]
            ),
            Gain(echo_params["gain_adjustment"]),
        ]
    )


def create_vocal_chain(
    voc_input: float,
    release: float = 300,
    feedback: float = 180,
    genre: MusicGenre = MusicGenre.POP,
    voice_type: VoiceType = VoiceType.FEMALE,
    deesser_strength: DeEsserStrength = DeEsserStrength.MODERATE,
    compression_strength: CompressionStrength = CompressionStrength.MODERATE,
    eq_style: EQStyle = EQStyle.NEUTRAL,
    echo_level: EchoLevel = EchoLevel.OFF,
    basic_time: float = 500,
) -> Pedalboard:
    """创建人声处理链"""

    # 获取各种参数
    genre_params = get_genre_parameters(genre)
    voice_params = get_voice_type_parameters(voice_type)
    deesser_params = get_deesser_parameters(deesser_strength)
    eq_params = get_eq_style_parameters(eq_style)

    # 压缩比例调整
    compression_ratios = {
        CompressionStrength.LIGHT: 1.8,
        CompressionStrength.MODERATE: 2.5,
        CompressionStrength.HEAVY: 3.5,
    }
    base_ratio = compression_ratios[compression_strength]
    final_ratio = base_ratio * (genre_params["compression_ratio"] / 2.5)

    effects = [
        Gain(voc_input),
        HighpassFilter(voice_params["highpass_freq"]),
    ]

    # 基础EQ
    if eq_params["low_shelf_gain"] != 0:
        effects.append(LowShelfFilter(200, eq_params["low_shelf_gain"], 0.7))

    effects.extend(
        [
            PeakFilter(
                2700 + voice_params["presence_freq"] - 2500,
                -2 + eq_params["mid_gain"],
                1,
            ),
            HighShelfFilter(
                20000,
                -2 + eq_params["high_shelf_gain"] + genre_params["high_shelf_gain"],
                1.8,
            ),
            Gain(1),
            PeakFilter(
                voice_params["presence_freq"],
                voice_params["presence_gain"] + genre_params["vocal_brightness"],
                1.15,
            ),
            PeakFilter(
                voice_params["brightness_freq"],
                voice_params["brightness_gain"] + genre_params["vocal_brightness"],
                1,
            ),
        ]
    )

    # 去齿音处理
    if deesser_strength != DeEsserStrength.OFF:
        effects.append(
            PeakFilter(
                deesser_params["freq"], deesser_params["gain"], deesser_params["q"]
            )
        )

    # 回声处理
    if echo_level != EchoLevel.OFF:
        echo_chain = create_echo_chain(echo_level, basic_time)
        effects.append(echo_chain)

    effects.extend(
        [
            Gain(-1),
            Mix(
                [
                    Gain(0),
                    Pedalboard(
                        [Invert(), Compressor(-30, 3.2, 40, feedback), Gain(-40)]
                    ),
                ]
            ),
            Compressor(-18, final_ratio, 19, release),
            Gain(0),
        ]
    )

    return Pedalboard(effects)


def get_reverb_parameters(level: ReverbLevel) -> Tuple[float, float, float, float]:
    """根据混响等级获取混响参数

    Args:
        level: 混响等级

    Returns:
        (room_size, damping, wet_level, reverb_gain_adjustment) 参数元组
    """
    parameters = {
        ReverbLevel.DRY: (0.0, 1.0, 0.0, -60),  # 无混响
        ReverbLevel.SUBTLE: (0.2, 0.8, 0.1, -18),  # 微妙混响
        ReverbLevel.LIGHT: (0.3, 0.7, 0.2, -12),  # 轻度混响
        ReverbLevel.MODERATE: (0.5, 0.6, 0.3, -6),  # 中度混响（默认）
        ReverbLevel.HEAVY: (0.7, 0.4, 0.5, 0),  # 重度混响
        ReverbLevel.EXTREME: (0.9, 0.2, 0.7, 3),  # 极重混响
    }
    return parameters[level]


def create_reverb_chain(
    reverb_gain: float,
    short: float = 5,
    medium: float = 25,
    long: float = 50,
    delay: float = 200,
    level: ReverbLevel = ReverbLevel.MODERATE,
) -> Pedalboard:
    """创建混响处理链"""
    room_size, damping, wet_level, gain_adjustment = get_reverb_parameters(level)

    # 如果是干声模式，返回一个只有增益的简单链
    if level == ReverbLevel.DRY:
        return Pedalboard([Gain(reverb_gain + gain_adjustment)])

    # 根据混响等级调整参数
    adjusted_reverb_gain = reverb_gain + gain_adjustment

    delay_chain = Pedalboard(
        [
            Gain(-20),
            Delay(delay / 8, 0, wet_level * 0.3),
            Gain(-12),
        ]
    )

    short_reverb = Pedalboard(
        [
            Gain(-20),
            Delay(short / 1000, 0, wet_level * 0.2),
            Reverb(room_size * 0.4, damping, wet_level, 0, 1, 0),
            Gain(-12),
        ]
    )

    medium_reverb = Pedalboard(
        [
            Gain(-16),
            Delay(medium / 1000, 0.3, wet_level * 0.4),
            Reverb(room_size * 0.7, damping, wet_level, 0, 1, 0),
            Gain(-19),
        ]
    )

    long_reverb = Pedalboard(
        [
            Gain(-12),
            Delay(long / 1000, 0.6, wet_level * 0.6),
            Reverb(room_size, damping, wet_level, 0, 1, 0),
            Gain(-23),
        ]
    )

    return Pedalboard(
        [
            Mix([short_reverb, medium_reverb, long_reverb, delay_chain]),
            PeakFilter(1450, -4, 1.83),
            PeakFilter(2300, 5, 0.51),
            Gain(adjusted_reverb_gain),
        ]
    )


def create_instrument_chain(headroom: float) -> Pedalboard:
    """创建乐器处理链"""
    return Pedalboard([Gain(headroom)])


def create_master_chain(
    comp_release: float = 500, limiter_release: float = 400
) -> Pedalboard:
    """创建总线处理链"""
    return Pedalboard(
        [
            Compressor(-10, 1.6, 10, comp_release),
            Limiter(-3, limiter_release),
            Gain(-0.5),
        ]
    )


def combine_audio(
    vocal: np.ndarray, reverb: np.ndarray, instrument: np.ndarray
) -> np.ndarray:
    """合并音频轨道"""
    min_length = min(vocal.shape[1], instrument.shape[1])
    vocal_trimmed = vocal[:, :min_length]
    reverb_trimmed = reverb[:, :min_length]
    instrument_trimmed = instrument[:, :min_length]

    return vocal_trimmed + instrument_trimmed + reverb_trimmed


def write_audio(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """写入音频文件"""
    sf.write(path, audio.T, samplerate=sample_rate, format="flac")


def automix(
    voc_path: str,
    inst_path: str,
    sample_rate: int = 44100,
    reverb_gain: int = 0,
    headroom: int = -8,
    voc_input: int = -4,
    reverb_level: ReverbLevel = ReverbLevel.MODERATE,
    music_genre: MusicGenre = MusicGenre.POP,
    voice_type: VoiceType = VoiceType.FEMALE,
    deesser_strength: DeEsserStrength = DeEsserStrength.MODERATE,
    compression_strength: CompressionStrength = CompressionStrength.MODERATE,
    eq_style: EQStyle = EQStyle.NEUTRAL,
    echo_level: EchoLevel = EchoLevel.OFF,
) -> str:
    """自动混音处理

    Args:
        voc_path: 人声文件路径
        inst_path: 伴奏文件路径
        sample_rate: 采样率
        reverb_gain: 混响增益
        headroom: 乐器动态余量
        voc_input: 人声输入增益
        reverb_level: 混响等级
        music_genre: 音乐风格
        voice_type: 人声类型
        deesser_strength: 去齿音强度
        compression_strength: 压缩强度
        eq_style: EQ风格
        echo_level: 回声等级

    Returns:
        输出文件路径
    """
    # 计算时间参数
    time_calculator = TimeCalculator(inst_path)
    pre_delay = time_calculator.times["pre_delay"]
    release = time_calculator.times["release"]

    # 加载音频
    vocal_audio = load_audio(voc_path, sample_rate)
    instrument_audio = load_audio(inst_path, sample_rate)

    # 根据风格调整混响增益
    genre_params = get_genre_parameters(music_genre)
    adjusted_reverb_gain = reverb_gain + genre_params["reverb_adjustment"]

    # 创建处理链
    vocal_fx = create_vocal_chain(
        voc_input,
        release[1],
        release[0],
        music_genre,
        voice_type,
        deesser_strength,
        compression_strength,
        eq_style,
        echo_level,
        time_calculator.basic_time,
    )
    reverb_fx = create_reverb_chain(
        adjusted_reverb_gain,
        pre_delay[0],
        pre_delay[2],
        pre_delay[3],
        pre_delay[1],
        reverb_level,
    )
    instrument_fx = create_instrument_chain(headroom)
    master_fx = create_master_chain(release[3], release[2])

    # 处理人声
    processed_vocal = vocal_fx(vocal_audio, sample_rate)

    # 处理立体声
    if processed_vocal.ndim > 1 and processed_vocal.shape[0] > 2:
        stereo = np.mean(processed_vocal, axis=0, keepdims=True)
    else:
        stereo = processed_vocal

    # 处理混响和乐器
    reverb_audio = reverb_fx(stereo, sample_rate)
    processed_instrument = instrument_fx(instrument_audio, sample_rate)

    # 合并音频
    combined_audio = combine_audio(processed_vocal, reverb_audio, processed_instrument)

    # 主总线处理
    final_output = master_fx(combined_audio, sample_rate)

    # 输出文件
    output_path = tempfile.mktemp(suffix=".flac")
    write_audio(output_path, final_output, sample_rate)

    return output_path
