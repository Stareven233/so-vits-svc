r'''
cd D:/Code/projects/so-vits-svc
$python = 'D:/Code/projects/RIFT-SVC/.venv/Scripts/python.exe'

$name = 'aino'
$name = 'fritia'
$name = 'megumin'
$name = '「少女」'

1.
$src_dir = "D:\Document\Audio\$name"
& $python util/fap/main.py slice-audio-v2 $src_dir "data/$name" --max-duration 15.0 --num-workers 2 --flat-layout --merge-short

2.
& $python -m preprocess1_config -n $name --source_dir 'data' --speech_encoder vec768l12 --vol_aug --speakers $name

3.
& $python -m preprocess2_feature --f0_predictor fcpe --filelist filelists/train.txt --num_workers 2
& $python -m preprocess2_feature --f0_predictor fcpe --filelist filelists/val.txt --num_workers 2

4.
D:/Code/projects/RIFT-SVC/.venv/Scripts/python.exe -m train -m fritia -p bf16 -e 350 -bs 14 --eval_interval 1000 --torch_compile_mode default
& $python -m train -m $name -p bf16 -e 350 -bs 14 --eval_interval 1000 --torch_compile_mode default
& $python -m train -m $name -p bf16 -e 120 -bs 14 --all_in_mem

5.
tensorboard --logdir "exp/$name"

6.
$path = "D:/Document/ai-sings/新月的摇篮曲 (其一)  伴月同眠/哥伦比娅 伴月同眠 - 测试服废案_Vocals_vocals_noreverb.flac"
$path = "D:\Document\ai-sings\春庭雪\4k无损春庭雪橙翼_Vocals_vocals_noreverb.flac"
& $python infer.py -m exp/$name -i $path -t 4


$python = 'D:/Code/projects/RIFT-SVC/.venv/Scripts/python.exe'
cd D:/Code/projects/so-vits-svc
$name = '「少女」'
$indir = 'D:/Document/ai-sings'
$path = "${indir}/God Knows/4K高清修复音源升级God Knows_Vocals_vocals_noreverb-new-au.flac"
$path = "${indir}/TAIDADA/TAIDADA_反相不纯人声_Vocals_vocals_noreverb.flac"
$path = "$indir\心愿便利贴\心愿便利贴-王欣宇_vocals_noreverb.flac"
$path = "$indir\ツキアカリのミチシルベ\4K 60FPS黑之契约者 流星的双子 stereopony月光的指引_Vocals_vocals.flac"
$path = "${indir}/君は薔薇より美しい/君は薔薇より美しい_呼!.flac"
& $python infer.py -m exp/$name -i $path -t 0

New-Item -Path 'F:/CODE/!projects/so-vits-svc/pretrain/contentvec/checkpoint_best_legacy_500.pt' -ItemType HardLink -Target 'F:/CODE/!projects/DDSP-SVC/pretrain/contentvec/checkpoint_best_legacy_500.pt'
New-Item -Path 'F:/CODE/!projects/so-vits-svc/pretrain/rmvpe/model.pt' -ItemType HardLink -Target 'F:/CODE/!projects/DDSP-SVC/pretrain/rmvpe/model.pt'
'''
import logging
import argparse
from pathlib import Path
import re

import soundfile
import torch
import fairseq

torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])

from inference import infer_tool
from inference.infer_tool import Svc
from spkmix import spk_mix_map

# from loguru import logger
from util import logger

logging.getLogger('numba').setLevel(logging.WARNING)
ckpt_step_patten = re.compile(r'(?<=G_)\d+')  # G_16800.pth


def getLastestCheckpoint(m_dir: Path):
  files = tuple(f for f in m_dir.iterdir() if f.suffix == '.pth' and f.stem.startswith('G_'))
  if len(files) == 0:
    logger.error(f'no checkpoint in {m_dir}')
    return None
  latest_file = max(files)
  return latest_file


def main():
  parser = argparse.ArgumentParser(description='sovits4 inference')

  # 一定要设置的部分
  parser.add_argument('-m', '--model_path', type=Path, required=True, help='模型路径/目录')
  parser.add_argument(
      '-c',
      '--config_path',
      type=str,
      default=None,
      help='配置文件路径，默认为model文件夹下的config.json',
  )
  parser.add_argument(
      '-t',
      '--trans',
      type=int,
      nargs='+',
      default=[0],
      help='音高调整，支持正负（半音）',
  )
  parser.add_argument(
      '-s',
      '--spk_list',
      type=str,
      nargs='+',
      default=None,
      help='合成目标说话人名称，默认取配置中第一个说话人',
  )

  # 可选项部分
  parser.add_argument(
      '-i',
      '--input',
      type=str,
      nargs='+',
      default=[],
      help='输入文件名列表',
  )
  parser.add_argument(
      '-cl',
      '--clip',
      type=float,
      default=0,
      help='音频强制切片，默认0为自动切片，单位为秒/s',
  )
  parser.add_argument(
      '-a',
      '--auto_predict_f0',
      action='store_true',
      default=False,
      help='语音转换自动预测音高，转换歌声时不要打开这个会严重跑调',
  )
  parser.add_argument(
      '-cm',
      '--cluster_model_path',
      type=str,
      default='',
      help='聚类模型或特征检索索引路径，留空则自动设为各方案模型的默认路径，如果没有训练聚类或特征检索则随便填',
  )
  parser.add_argument(
      '-cr',
      '--cluster_infer_ratio',
      type=float,
      default=0,
      help='聚类方案或特征检索占比，范围0-1，若没有训练聚类模型或特征检索则默认0即可',
  )
  parser.add_argument(
      '-lg',
      '--linear_gradient',
      type=float,
      default=0,
      help='两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，单位为秒',
  )
  parser.add_argument(
      '-f0p',
      '--f0_predictor',
      type=str,
      default='rmvpe',
      help='选择F0预测器,可选择crepe,pm,dio,harvest,rmvpe,fcpe默认为pm(注意：crepe为原F0使用均值滤波器)',
  )
  parser.add_argument(
      '-eh',
      '--enhance',
      action='store_true',
      default=False,
      help='是否使用NSF_HIFIGAN增强器,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭',
  )
  parser.add_argument(
      '-shd',
      '--shallow_diffusion',
      action='store_true',
      default=False,
      help='是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN增强器将会被禁止',
  )
  parser.add_argument(
      '-usm',
      '--use_spk_mix',
      action='store_true',
      default=False,
      help='是否使用角色融合',
  )
  parser.add_argument(
      '-lea',
      '--loudness_envelope_adjustment',
      type=float,
      default=1,
      help='输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络',
  )
  parser.add_argument(
      '-fr',
      '--feature_retrieval',
      action='store_true',
      default=False,
      help='是否使用特征检索，如果使用聚类模型将被禁用，且cm与cr参数将会变成特征检索的索引路径与混合比例',
  )

  # 浅扩散设置
  parser.add_argument(
      '-dm',
      '--diffusion_model_path',
      type=str,
      default='exp/default/diffusion/model_0.pt',
      help='扩散模型路径',
  )
  parser.add_argument(
      '-dc',
      '--config_diff_path',
      type=str,
      default=None,
      help='扩散模型配置文件路径，默认为model文件夹下的config_diff.yaml',
  )
  parser.add_argument(
      '-ks',
      '--k_step',
      type=int,
      default=100,
      help='扩散步数，越大越接近扩散模型的结果，默认100',
  )
  parser.add_argument(
      '-se',
      '--second_encoding',
      action='store_true',
      default=False,
      help='二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差',
  )
  parser.add_argument(
      '-od',
      '--only_diffusion',
      action='store_true',
      default=False,
      help='纯扩散模式，该模式不会加载sovits模型，以扩散模型推理',
  )

  # 不用动的部分
  parser.add_argument(
      '-sd',
      '--slice_db',
      type=int,
      default=-40,
      help='默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50',
  )
  parser.add_argument(
      '-d',
      '--device',
      type=str,
      default=None,
      help='推理设备，None则为自动选择cpu和gpu',
  )
  parser.add_argument(
      '-ns',
      '--noice_scale',
      type=float,
      default=0.4,
      help='噪音级别，会影响咬字和音质，较为玄学',
  )
  parser.add_argument(
      '-p',
      '--pad_seconds',
      type=float,
      default=0.5,
      help='推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现',
  )
  parser.add_argument('-wf', '--wav_format', type=str, default='flac', help='音频输出格式')
  parser.add_argument(
      '-lgr',
      '--linear_gradient_retain',
      type=float,
      default=0.75,
      help='自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭',
  )
  parser.add_argument(
      '-eak',
      '--enhancer_adaptive_key',
      type=int,
      default=0,
      help='使增强器适应更高的音域(单位为半音数)|默认为0',
  )
  parser.add_argument(
      '-ft',
      '--f0_filter_threshold',
      type=float,
      default=0.05,
      help='F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音',
  )
  parser.add_argument(
      '-v',
      '--vocal_register_shift_key',
      type=int,
      required=False,
      default=0,
      # 输入正值表示先降key进行推理再让声码器升回来
      help='音区偏移 (单位半音) , 只对 pc 声码器有效',
  )
  parser.add_argument(
      '--vocoder_type',
      type=str,
      required=False,
      default='nsf-hifigan',
      help='声码器类型',
  )
  parser.add_argument(
      '--vocoder_ckpt_path',
      type=str,
      required=False,
      default='pretrain/vocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt',
      help='声码器权重路径',
  )

  args = parser.parse_args()

  model_path = args.model_path
  trans = args.trans
  spk_list = args.spk_list
  slice_db = args.slice_db
  wav_format = args.wav_format
  auto_predict_f0 = args.auto_predict_f0
  cluster_infer_ratio = args.cluster_infer_ratio
  noice_scale = args.noice_scale
  pad_seconds = args.pad_seconds
  clip = args.clip
  lg = args.linear_gradient
  lgr = args.linear_gradient_retain
  f0p = args.f0_predictor
  enhance = args.enhance
  enhancer_adaptive_key = args.enhancer_adaptive_key
  cr_threshold = args.f0_filter_threshold
  diffusion_model_path = args.diffusion_model_path
  k_step = args.k_step
  only_diffusion = args.only_diffusion
  shallow_diffusion = args.shallow_diffusion
  use_spk_mix = args.use_spk_mix
  second_encoding = args.second_encoding
  loudness_envelope_adjustment = args.loudness_envelope_adjustment  # default: 1
  vocal_register_factor = 2**(args.vocal_register_shift_key / 12)

  if model_path.is_dir():
    model_path = getLastestCheckpoint(model_path)
    logger.info(f'Auto choose {model_path}')
  assert model_path.is_file(), f'非法模型权重: "{model_path}"'

  if (config_path := args.config_path) is None:
    config_path = model_path.with_name('config.json').as_posix()
  if (config_diff_path := args.config_diff_path) is None:
    config_diff_path = model_path.with_name('config_diff.yaml').as_posix()

  if cluster_infer_ratio != 0:
    if args.cluster_model_path == '':
      if (args.feature_retrieval):  # 若指定了占比但没有指定模型路径，则按是否使用特征检索分配默认的模型路径
        args.cluster_model_path = 'exp/default/feature_and_index.pkl'
      else:
        args.cluster_model_path = 'exp/default/kmeans_10000.pt'
  else:  # 若未指定占比，则无论是否指定模型路径，都将其置空以避免之后的模型加载
    args.cluster_model_path = ''

  svc_model = Svc(
      model_path,
      config_path,
      args.vocoder_type,
      args.vocoder_ckpt_path,
      args.device,
      args.cluster_model_path,
      enhance,
      diffusion_model_path,
      config_diff_path,
      shallow_diffusion,
      only_diffusion,
      use_spk_mix,
      args.feature_retrieval,
  )

  if len(spk_mix_map) <= 1:
    use_spk_mix = False
  if use_spk_mix:
    spk_list = [spk_mix_map]
  if spk_list is None:
    spk_list = tuple(svc_model.config.spk.keys())[:1]

  infer_tool.fill_a_to_b(trans, args.input)
  for in_file, tran in zip(args.input, trans):
    in_file = Path(in_file)

    if in_file.suffix == '':
      in_file = in_file.with_suffix('.wav')
    for spk in spk_list:
      audio = svc_model.slice_inference(
        infer_tool.format_wav_to_memory(in_file),
        spk,
        tran,
        slice_db,
        cluster_infer_ratio,
        auto_predict_f0,
        noice_scale,
        pad_seconds,
        clip, lg, lgr, f0p,
        enhancer_adaptive_key,
        cr_threshold,
        k_step=k_step,
        use_spk_mix=use_spk_mix,
        second_encoding=second_encoding,
        loudness_envelope_adjustment=loudness_envelope_adjustment,
        vocal_register_factor=vocal_register_factor,
      )
      key = '~' if auto_predict_f0 else f'{tran}'
      cluster_name = '' if cluster_infer_ratio == 0 else f'_{cluster_infer_ratio}'
      isdiffusion = 'sov'
      m = model_path.stem
      if shallow_diffusion:
        isdiffusion = 'sovdiff'
      if only_diffusion:
        isdiffusion = 'diff'
        m = diffusion_model_path
      if use_spk_mix:
        spk = 'spk_mix'
        # rift@「少女」_4.0ks_0k_-60.0st
      m = ckpt_step_patten.search(m)
      ks = int(m.group(0)) / 1000
      out_file = in_file.with_name(f'{in_file.stem}_{isdiffusion}@{spk}_{ks:.2f}ks_{key}k{cluster_name}_{args.vocal_register_shift_key}vk.{wav_format}')
      soundfile.write(out_file, audio, svc_model.target_sample, format=wav_format)
      svc_model.clear_empty()


if __name__ == '__main__':
  main()
