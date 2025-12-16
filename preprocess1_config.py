'''
根据模板生成配置文件，划分训练验证集，将文件名列表写入本地 ./filelists/xx.txt
'''

import argparse
import wave
from random import shuffle
import concurrent
from pathlib import Path

from util import io
from util import logger


def get_wav_duration(path):
    try:
        with wave.open(path, 'rb') as wav_file:
            # 获取音频帧数
            n_frames = wav_file.getnframes()
            # 获取采样率
            framerate = wav_file.getframerate()
            # 计算时长（秒）
            return n_frames / float(framerate), path
    except Exception as e:
        logger.error(f'Reading {path}')
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='default', help='exp/model name')
    parser.add_argument(
        '--train_list',
        type=str,
        default='./filelists/train.txt',
        help='path to train list',
    )
    parser.add_argument(
        '--val_list', type=str, default='./filelists/val.txt', help='path to val list'
    )
    parser.add_argument(
        '--source_dir', type=Path, default='./data', help='path to source dir'
    )
    parser.add_argument(
        '--speech_encoder',
        type=str,
        default='vec768l12',
        help='choice a speech encoder|\'vec768l12\',\'vec256l9\',\'hubertsoft\',\'whisper-ppg\',\'cnhubertlarge\',\'dphubert\',\'whisper-ppg-large\',\'wavlmbase+\'',
    )
    parser.add_argument(
        '--vol_aug',
        action='store_true',
        help='Whether to use volume embedding and volume augmentation',
    )
    parser.add_argument(
        '--tiny', action='store_true', help='Whether to train sovits tiny'
    )
    parser.add_argument(
        '--speakers',
        nargs='+',
        required=False,
        default=None,
        help='All speakers to be trained, default all subdir name in --source_dir'
    )
    args = parser.parse_args()

    config_dir = Path('configs/')
    config = io.load_json(
        config_dir / ('template/main_template_tiny.json' if args.tiny else 'template/main_template.json'),
    )
    train = []
    val = []
    idx = 0
    spk_dict = {}
    spk_id = 0
    if args.speakers is None:
        args.speakers = tuple(d.stem for d in args.source_dir.iterdir() if d.is_dir())
    with (
        logger.Progress() as progress,
        concurrent.futures.ThreadPoolExecutor() as executor,
    ):
        for speaker in progress.track(args.speakers, description='Processing Speakers'):
            spk_dict[speaker] = spk_id
            spk_id += 1
            wavs = []

            futures = []
            for file_path in (args.source_dir / speaker).iterdir():
                Path.suffix
                if file_path.stem.startswith('.') or file_path.suffix != '.wav':
                    continue

                file_path = file_path.as_posix()
                futures.append(executor.submit(get_wav_duration, file_path))

            for future in concurrent.futures.as_completed(futures):
                try:
                    duration, path = future.result()
                    if duration < 0.3:
                        logger.info('Skip too short audio')
                        continue
                    wavs.append(path)
                except Exception as e:
                    logger.error(f'Error processing file: {e}')

            shuffle(wavs)
            num_val = max(2, int(len(wavs) * 0.01))
            train.extend(wavs[num_val:])
            val.extend(wavs[:num_val])

        logger.info('Writing ' + args.train_list)
        with open(args.train_list, 'w', encoding='utf-8') as f:
            for fname in progress.track(train, description='Writing train list'):
                wavpath = fname
                f.write(wavpath + '\n')

        logger.info('Writing ' + args.val_list)
        with open(args.val_list, 'w', encoding='utf-8') as f:
            for fname in progress.track(val, description='Writing val list'):
                wavpath = fname
                f.write(wavpath + '\n')

        config_diff = io.load_yaml('configs/template/diffusion_template.yaml')
        config_diff['model']['n_spk'] = spk_id
        config_diff['data']['encoder'] = args.speech_encoder
        config_diff['spk'] = spk_dict

        config['spk'] = spk_dict
        config['model']['n_speakers'] = spk_id
        config['model']['speech_encoder'] = args.speech_encoder

        if (
            args.speech_encoder == 'vec768l12'
            or args.speech_encoder == 'dphubert'
            or args.speech_encoder == 'wavlmbase+'
        ):
            config['model']['ssl_dim'] = config['model'][
                'filter_channels'
            ] = config['model']['gin_channels'] = 768
            config_diff['data']['encoder_out_channels'] = 768
        elif args.speech_encoder == 'vec256l9' or args.speech_encoder == 'hubertsoft':
            config['model']['ssl_dim'] = config['model'][
                'gin_channels'
            ] = 256
            config_diff['data']['encoder_out_channels'] = 256
        elif (
            args.speech_encoder == 'whisper-ppg'
            or args.speech_encoder == 'cnhubertlarge'
        ):
            config['model']['ssl_dim'] = config['model'][
                'filter_channels'
            ] = config['model']['gin_channels'] = 1024
            config_diff['data']['encoder_out_channels'] = 1024
        elif args.speech_encoder == 'whisper-ppg-large':
            config['model']['ssl_dim'] = config['model'][
                'filter_channels'
            ] = config['model']['gin_channels'] = 1280
            config_diff['data']['encoder_out_channels'] = 1280

        if args.vol_aug:
            config['train']['vol_aug'] = config['model'][
                'vol_embedding'
            ] = True

        if args.tiny:
            config['model']['filter_channels'] = 512

        exp_dir = Path('exp', args.name)
        exp_dir.mkdir(exist_ok=True)
        io.write_json(exp_dir / 'config.json', config, indent=2)
        io.write_json(config_dir / 'config.json', config, indent=2)

        io.write_yaml(exp_dir / 'config_diff.yaml', config_diff)
        io.write_yaml(config_dir / 'config_diff.yaml', config_diff)
        logger.info(f'Writing configs to {config_dir} and {exp_dir}')
