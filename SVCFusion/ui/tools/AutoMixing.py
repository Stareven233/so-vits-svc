from util import Dummy; gr = Dummy()
from loguru import logger

from SVCFusion import automix
from SVCFusion.automix import (
    ReverbLevel,
    MusicGenre,
    VoiceType,
    DeEsserStrength,
    CompressionStrength,
    EQStyle,
    EchoLevel,
)
from SVCFusion.i18n import I
from SVCFusion.ui.component.Warning import WarningHTML


class AutoMixing:
    def __init__(self) -> None:
        gr.Markdown(
            f"""
## {I.auto_mixing.title}

{I.auto_mixing.description}

{I.auto_mixing.support_info}

> {I.auto_mixing.contributor_info}
            """
        )

        WarningHTML(I.auto_mixing.upload_warning)

        # 基础输入
        input_vocals = gr.Audio(label=I.auto_mixing.input_vocals_label, type="filepath")
        input_bgm = gr.Audio(label=I.auto_mixing.input_bgm_label, type="filepath")

        # 音量调节
        with gr.Row():
            vocal_volume = gr.Slider(
                label="人声音量",
                minimum=-20,
                maximum=20,
                value=0,
                step=0.5,
                info="调整人声音量 (dB)",
            )

        # 音乐风格预设
        music_genre = gr.Dropdown(
            label=I.auto_mixing.music_genre_label,
            choices=[
                (I.auto_mixing.music_genre_pop, "POP"),
                (I.auto_mixing.music_genre_rock, "ROCK"),
                (I.auto_mixing.music_genre_jazz, "JAZZ"),
                (I.auto_mixing.music_genre_electronic, "ELECTRONIC"),
                (I.auto_mixing.music_genre_folk, "FOLK"),
                (I.auto_mixing.music_genre_classical, "CLASSICAL"),
            ],
            value="POP",
            info=I.auto_mixing.music_genre_info,
        )

        # 人声类型优化
        voice_type = gr.Dropdown(
            label=I.auto_mixing.voice_type_label,
            choices=[
                (I.auto_mixing.voice_type_male_low, "MALE_LOW"),
                (I.auto_mixing.voice_type_male_high, "MALE_HIGH"),
                (I.auto_mixing.voice_type_female, "FEMALE"),
                (I.auto_mixing.voice_type_rap, "RAP"),
                (I.auto_mixing.voice_type_vocal, "VOCAL"),
            ],
            value="FEMALE",
            info=I.auto_mixing.voice_type_info,
        )

        # 高级选项折叠面板
        with gr.Accordion(I.auto_mixing.advanced_options_title, open=False):
            gr.Markdown(f"### {I.auto_mixing.professional_params_title}")

            with gr.Row():
                # 去齿音强度
                deesser_strength = gr.Dropdown(
                    label=I.auto_mixing.deesser_strength_label,
                    choices=[
                        (I.auto_mixing.deesser_strength_off, "OFF"),
                        (I.auto_mixing.deesser_strength_light, "LIGHT"),
                        (I.auto_mixing.deesser_strength_moderate, "MODERATE"),
                        (I.auto_mixing.deesser_strength_heavy, "HEAVY"),
                    ],
                    value="MODERATE",
                    info=I.auto_mixing.deesser_strength_info,
                )

                # 压缩强度
                compression_strength = gr.Dropdown(
                    label=I.auto_mixing.compression_strength_label,
                    choices=[
                        (I.auto_mixing.compression_strength_light, "LIGHT"),
                        (I.auto_mixing.compression_strength_moderate, "MODERATE"),
                        (I.auto_mixing.compression_strength_heavy, "HEAVY"),
                    ],
                    value="MODERATE",
                    info=I.auto_mixing.compression_strength_info,
                )

            with gr.Row():
                # EQ风格
                eq_style = gr.Dropdown(
                    label=I.auto_mixing.eq_style_label,
                    choices=[
                        (I.auto_mixing.eq_style_neutral, "NEUTRAL"),
                        (I.auto_mixing.eq_style_bright, "BRIGHT"),
                        (I.auto_mixing.eq_style_warm, "WARM"),
                        (I.auto_mixing.eq_style_vintage, "VINTAGE"),
                    ],
                    value="NEUTRAL",
                    info=I.auto_mixing.eq_style_info,
                )

                # 混响等级
                reverb_level = gr.Dropdown(
                    label=I.auto_mixing.reverb_level_label,
                    choices=[
                        (I.auto_mixing.reverb_level_dry, "DRY"),
                        (I.auto_mixing.reverb_level_subtle, "SUBTLE"),
                        (I.auto_mixing.reverb_level_light, "LIGHT"),
                        (I.auto_mixing.reverb_level_moderate, "MODERATE"),
                        (I.auto_mixing.reverb_level_heavy, "HEAVY"),
                        (I.auto_mixing.reverb_level_extreme, "EXTREME"),
                    ],
                    value="MODERATE",
                    info=I.auto_mixing.reverb_level_info,
                )

                # 回声等级
                echo_level = gr.Dropdown(
                    label=I.auto_mixing.echo_level_label,
                    choices=[
                        (I.auto_mixing.echo_level_off, "OFF"),
                        (I.auto_mixing.echo_level_light, "LIGHT"),
                        (I.auto_mixing.echo_level_moderate, "MODERATE"),
                        (I.auto_mixing.echo_level_heavy, "HEAVY"),
                    ],
                    value="OFF",
                    info=I.auto_mixing.echo_level_info,
                )

        submit_btn = gr.Button(
            I.auto_mixing.submit_btn_value, variant="primary", size="lg"
        )

        output_mixed = gr.Audio(label=I.auto_mixing.output_mixed_label, type="filepath")

        submit_btn.click(
            self.process,
            inputs=[
                input_vocals,
                input_bgm,
                vocal_volume,
                music_genre,
                voice_type,
                deesser_strength,
                compression_strength,
                eq_style,
                reverb_level,
                echo_level,
            ],
            outputs=output_mixed,
        )

    def process(
        self,
        input_vocals: str,
        input_bgm: str,
        vocal_volume: float,
        music_genre_str: str,
        voice_type_str: str,
        deesser_strength_str: str,
        compression_strength_str: str,
        eq_style_str: str,
        reverb_level_str: str,
        echo_level_str: str,
    ):
        logger.info(I.auto_mixing.processing_start_log)
        logger.info(f"人声音量: {vocal_volume}dB")
        logger.info(f"音乐风格: {music_genre_str}, 人声类型: {voice_type_str}")
        logger.info(f"去齿音: {deesser_strength_str}, 压缩: {compression_strength_str}")
        logger.info(
            f"EQ风格: {eq_style_str}, 混响等级: {reverb_level_str}, 回声等级: {echo_level_str}"
        )

        # 将字符串转换为对应的枚举
        music_genre = getattr(MusicGenre, music_genre_str)
        voice_type = getattr(VoiceType, voice_type_str)
        deesser_strength = getattr(DeEsserStrength, deesser_strength_str)
        compression_strength = getattr(CompressionStrength, compression_strength_str)
        eq_style = getattr(EQStyle, eq_style_str)
        reverb_level = getattr(ReverbLevel, reverb_level_str)
        echo_level = getattr(EchoLevel, echo_level_str)

        result = automix.automix(
            voc_path=input_vocals,
            inst_path=input_bgm,
            sample_rate=44100,
            reverb_gain=0,
            headroom=-8,
            voc_input=-4 + vocal_volume,
            reverb_level=reverb_level,
            music_genre=music_genre,
            voice_type=voice_type,
            deesser_strength=deesser_strength,
            compression_strength=compression_strength,
            eq_style=eq_style,
            echo_level=echo_level,
        )
        logger.info(I.auto_mixing.processing_complete_log)
        return result
