import os

from tqdm import tqdm

from SVCFusion.config import JSONReader, YAMLReader
from SVCFusion.model_utils import detect_current_model_by_path
from SVCFusion.exec import executable
from loguru import logger

from SVCFusion.file import make_dirs
from SVCFusion.exec import exec


class DrawArgs:
    val = "data/val/audio"
    sample_rate = 1
    train = "data/train/audio"
    extensions = ["wav", "flac"]


class PreprocessArgs:
    config = "configs/ddsp_reflow.yaml"
    device = "cuda"


def resample(src, dst):
    assert (
        exec(
            f'{executable} fap/__main__.py resample "{src}" "{dst}" --mono',
        )
        == 0
    ), "重采样失败，请截图日志反馈，日志在上面 不在这里！！"


def to_wav(src, dst):
    assert (
        exec(
            f'{executable} fap/__main__.py to-wav "{src}" "{dst}"',
        )
        == 0
    ), "转 WAV 失败，请截图日志反馈，日志在上面 不在这里！！"


def slice_audio(
    src: str,
    dst: str,
    max_duration: float,
    max_workers: int = 1,
):
    cmd = (
        f"{executable} fap/__main__.py slice-audio-v2 "
        f'"{src}" "{dst}" --max-duration {max_duration} '
        f"--num-workers {max_workers} "
        "--flat-layout --merge-short --clean"
    )
    result = exec(cmd)
    assert result == 0, "切割音频失败，请截图日志反馈，日志在上面不在这里！！"


def auto_normalize_dataset(
    output_dir: str,
    rename_by_index: bool,
    # _progress: gr.Progress,  # 我为什么要引入这个傻逼东西
    max_workers: int = os.cpu_count() or 1,
    skip_slice: bool = False,
):
    spks = get_spks_from_dataset_raw()

    # 扫描角色目录，如果发现 .WAV 文件 改成 .wav
    for spk in spks:
        spk_path = os.path.join("dataset_raw", spk)
        if not os.path.isdir(spk_path):
            continue
        for file in os.listdir(spk_path):
            if file.endswith(".WAV"):
                old_path = os.path.join(spk_path, file)
                new_path = os.path.join(spk_path, file[:-4] + ".wav")
                logger.info(f"Renaming {old_path} to {new_path}")
                os.rename(old_path, new_path)

    make_dirs(output_dir, True)

    if not skip_slice:
        slice_audio(
            src="dataset_raw/",
            dst=output_dir,
            max_duration=15.0,
            max_workers=max_workers,
        )
    else:
        # 如果跳过切片，直接复制文件
        import shutil

        for spk in spks:
            src_spk_path = os.path.join("dataset_raw", spk)
            dst_spk_path = os.path.join(output_dir, spk)

            if not os.path.isdir(src_spk_path):
                continue

            make_dirs(dst_spk_path, True)

            for file in tqdm(os.listdir(src_spk_path)):
                if file.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                    src_file = os.path.join(src_spk_path, file)
                    dst_file = os.path.join(dst_spk_path, file)
                    shutil.copy2(src_file, dst_file)

    if rename_by_index:
        for i, spk in enumerate(
            [i for i in os.listdir(output_dir) if i != ".ipynb_checkpoints"]
        ):
            os.rename(f"{output_dir}/{spk}", f"{output_dir}/{i + 1}")


def get_spks_from_dataset_raw():
    spks = []
    for f in os.listdir("dataset_raw"):
        # if start with '.' => skip
        if f.startswith("."):
            logger.warning(f"Skip hidden file: {f}")
            continue

        if os.path.isdir(os.path.join("dataset_raw", f)):
            spks.append(f)
    return spks


def get_spk_from_dir(search_path):
    model_type_index = detect_current_model_by_path(search_path)
    if model_type_index == 2:
        with JSONReader(f"{search_path}/config.json") as config:
            return list(config["spk"].keys())
    elif model_type_index in [0, 1, 3, 4]:
        with YAMLReader(f"{search_path}/config.yaml") as config:
            return config["spks"]
