import argparse
import logging
import multiprocessing
import os
import time
from pathlib import Path
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rich import progress
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import (
  MultiPeriodDiscriminator,
  SynthesizerTrn,
)
from util import sov_utils as utils
from util.logger import Progress
from util.io import load_json, write_json
from util import Config
from modules import commons
from modules.losses import (
  discriminator_loss,
  feature_loss,
  generator_loss,
  kl_loss,
)
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def handle_configs(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-c',
    '--config',
    type=Path,
    default=None,
    help='JSON file for configuration',
  )
  parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
  parser.add_argument(
    '-p',
    '--precision',
    type=str,
    default=None,
    choices=('fp32', 'fp16', 'bf16', ),
    help='amp_dtype',
  )
  parser.add_argument('-e', '--epochs', type=int, default=None)
  parser.add_argument('--warmup_epochs', type=int, default=None)
  parser.add_argument('-lr', '--learning_rate', type=float, default=None)
  parser.add_argument('-bs', '--batch_size', type=int, default=None)
  parser.add_argument('--n_gpus', type=int, default=None)
  parser.add_argument('--num_workers', type=int, default=None)
  parser.add_argument('--eval_interval', type=int, default=None)
  parser.add_argument('--all_in_mem', action='store_true', default=False, help='加载所有数据集到内存中')
  parser.add_argument(
    '--pretrained_path', type=Path, default=Path('pretrain/sovits4.1'), 
    help='预训练模型路径，设空则不使用，且优先加载训练目录的已有权重。要求里面的预训练权重遵循 (G|D)_\\d+.pth 的格式'
  )
  parser.add_argument(
    '--torch_compile_mode', type=str, default=None,
    choices=('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs', ),
    help='Windows下需安装Visual Studio Tools 并在 Command Prompt for VS 20xx 中运行；首次编译很慢（max-autotune 10min左右）且需要更多显存'
  )

  args = parser.parse_args()
  model_dir = Path('./exp', args.model)
  if init and args.config is None:
    config_path = model_dir / 'config.json'
  else:
    config_path = args.config
  config = Config(load_json(config_path))

  config.train.update({k: v for k, v in vars(args).items() if v is not None})
  config.model_dir = model_dir.as_posix()
  if args.precision is not None:
    config.train.update({
      'fp16_run': args.precision.endswith('16'),
      'half_type': args.precision,
    })

  write_json(config_path, config.as_dict(), indent=2)
  return config


def main():
  '''Assume Single Node Multi GPUs Training Only'''
  assert torch.cuda.is_available(), 'CPU training is not allowed.'
  config = handle_configs()

  n_gpus = min(config.get('n_gpus', torch.cuda.device_count()), torch.cuda.device_count())
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = config.train.port

  mp.spawn(
    run,
    nprocs=n_gpus,
    args=(n_gpus, config, ),
  )


def save_checkpoints(g, og, d, od, c: Config, e: int):
  utils.save_checkpoint(
    g, og,
    c.train.learning_rate,
    e,
    os.path.join(c.model_dir, f'G_{global_step}.pth'),
  )
  utils.save_checkpoint(
    d, od,
    c.train.learning_rate,
    e,
    os.path.join(c.model_dir, f'D_{global_step}.pth'),
  )


def run(rank, n_gpus, hps: Config):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.hps(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, 'eval'))

  # for pytorch on win, backend use gloo
  dist.init_process_group(
    backend='gloo' if os.name == 'nt' else 'nccl',
    init_method='env://',
    world_size=n_gpus,
    rank=rank,
  )
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)
  collate_fn = TextAudioCollate()
  all_in_mem = hps.train.all_in_mem  # If you have enough memory, turn on this option to avoid disk IO and speed up training.
  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
  num_workers = (hps.train.num_workers if hasattr(hps.train, 'num_workers') else 2 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count())
  compile_mode = hps.train.torch_compile_mode
  if all_in_mem:
    num_workers = 1
  train_loader = DataLoader(
    train_dataset,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    drop_last=compile_mode is not None,
    batch_size=hps.train.batch_size,
    collate_fn=collate_fn,
  )
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, all_in_mem=all_in_mem, vol_aug=False)
    eval_loader = DataLoader(
      eval_dataset,
      num_workers=1,
      shuffle=False,
      batch_size=1,
      pin_memory=False,
      drop_last=False,
      collate_fn=collate_fn,
    )
  scaler = GradScaler(enabled=hps.train.fp16_run)

  net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
  ).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
    net_g.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
    fused=not scaler.is_enabled(),
  )
  optim_d = torch.optim.AdamW(
    net_d.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
    fused=not scaler.is_enabled(),
  )
  net_g = DDP(net_g, device_ids=[rank])  # , find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank])

  skip_optimizer = False
  try:
    _, _, _, last_epoch = utils.load_checkpoint(
      utils.latest_checkpoint_path(hps.model_dir, hps.train.pretrained_path, 'G_*.pth'),
      net_g,
      optim_g,
      skip_optimizer,
    )
    name = utils.latest_checkpoint_path(hps.model_dir, hps.train.pretrained_path, 'D_*.pth')
    utils.load_checkpoint(
      name,
      net_d,
      optim_d,
      skip_optimizer,
    )
    last_epoch = max(last_epoch, 1)
    global_step = int(name[name.rfind('_') + 1:name.rfind('.')]) + 1
    # global_step = (last_epoch - 1) * len(train_loader)
  except Exception as e:
    print(f'Warning: {e}, load old checkpoint failed... ')
    last_epoch = 1
    global_step = 0
    raise e
  if skip_optimizer:
    last_epoch = 1
    global_step = 0

  warmup_epoch = hps.train.warmup_epochs
  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=last_epoch - 2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=last_epoch - 2)

  if compile_mode is not None:
    logger.info('You are using [green]torch.compile[/green] for faster speed...')
    logger.info('Compiling the generator and discriminator...')
    net_g = torch.compile(net_g, mode=compile_mode)
    net_d = torch.compile(net_d, mode=compile_mode)
    # 在训练开始前编译优化器的step函数 optim_g optim_d, wait for muon
    # optimizer.step = torch.compile(optimizer.step, mode=compile_mode)
    logger.info('Compiled net_g and net_d!')

  print(f'training: {len(train_loader)} steps per epoch')
  with Progress() as progress:
    for epoch in range(last_epoch, hps.train.epochs + 1):
      # set up warm-up learning rate
      if epoch <= warmup_epoch:
        for param_group in optim_g.param_groups:
          param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
        for param_group in optim_d.param_groups:
          param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
      # training
      try:
        train_and_evaluate(
          rank,
          epoch,
          hps,
          (net_g, net_d),
          (optim_g, optim_d),
          # (scheduler_g, scheduler_d),
          scaler,
          (train_loader, eval_loader),
          logger,
          (writer, writer_eval),
          progress,
        )
      except KeyboardInterrupt:
        print('\n检测到 Ctrl+C，尝试在退出前保存权重...')
        save_checkpoints(net_g, optim_g, net_d, optim_d, hps, epoch)
        sys.exit()
      except Exception:
        err_log = os.path.join(hps.model_dir, 'error.log')
        with open(err_log, 'w', encoding='utf-8') as f:
          traceback.print_exc(file=f)
        save_checkpoints(net_g, optim_g, net_d, optim_d, hps, epoch)
        raise

      # update learning rate
      scheduler_g.step()
      scheduler_d.step()

  print('\n保存最后一个epoch权重...')
  save_checkpoints(net_g, optim_g, net_d, optim_d, hps, epoch)


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    scaler,
    loaders,
    logger,
    writers,
    progress: progress.Progress,
):
  net_g, net_d = nets
  optim_g, optim_d = optims
  train_loader, eval_loader = loaders
  if rank != 0:
    eval_loader = None
    logger = None
    writers = None
  if writers is not None:
    writer, writer_eval = writers

  half_type = torch.bfloat16 if hps.train.half_type == 'bf16' else torch.float16

  # train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  enumerated_train_loader = enumerate(train_loader)
  # logger.info(f'enumerated_train_loader len: {len(enumerated_train_loader)}')
  # 'Epoch {}'.format(epoch)
  task = progress.add_task(f'Epoch {epoch}', total=len(train_loader))
  for batch_idx, items in enumerated_train_loader:
    # logger.info(f'finish {progress.} ')
    c, f0, spec, y, spk, lengths, uv, volume = items
    g = spk.cuda(rank, non_blocking=True)
    spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
    c = c.cuda(rank, non_blocking=True)
    f0 = f0.cuda(rank, non_blocking=True)
    uv = uv.cuda(rank, non_blocking=True)
    lengths = lengths.cuda(rank, non_blocking=True)
    mel = spec_to_mel_torch(
      spec,
      hps.data.filter_length,
      hps.data.n_mel_channels,
      hps.data.sampling_rate,
      hps.data.mel_fmin,
      hps.data.mel_fmax,
    )

    with autocast('cuda', enabled=hps.train.fp16_run, dtype=half_type):
      (
        y_hat,
        ids_slice,
        z_mask,
        (z, z_p, m_p, logs_p, m_q, logs_q),
        pred_lf0,
        norm_lf0,
        lf0,
      ) = net_g(
        c, f0, uv, spec,
        g=g,
        c_lengths=lengths,
        spec_lengths=lengths,
        vol=volume,
      )

      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
      )
      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice
      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
    loss_disc_all = loss_disc

    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast('cuda', enabled=hps.train.fp16_run, dtype=half_type):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, losses_gen = generator_loss(y_d_hat_g)
    loss_lf0 = (F.mse_loss(pred_lf0, lf0) if net_g.module.use_automatic_f0_prediction else 0)
    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank == 0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
        reference_loss = 0
        for i in losses:
          reference_loss += i
        logger.info('Train Epoch: {} [{:.0f}%]'.format(epoch, 100.0 * batch_idx / len(train_loader)))
        logger.info(f'Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}')

        scalar_dict = {
          'loss/g/total': loss_gen_all,
          'loss/d/total': loss_disc_all,
          'learning_rate': lr,
          'grad_norm_d': grad_norm_d,
          'grad_norm_g': grad_norm_g,
        }
        scalar_dict.update({
          'loss/g/fm': loss_fm,
          'loss/g/mel': loss_mel,
          'loss/g/kl': loss_kl,
          'loss/g/lf0': loss_lf0,
        })

        # scalar_dict.update({'loss/g/{}'.format(i): v for i, v in enumerate(losses_gen)})
        # scalar_dict.update({'loss/d_r/{}'.format(i): v for i, v in enumerate(losses_disc_r)})
        # scalar_dict.update({'loss/d_g/{}'.format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = {
          'slice/mel_org': utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
          'slice/mel_gen': utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
          'all/mel': utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }

        if net_g.module.use_automatic_f0_prediction:
          image_dict.update({
            'all/lf0': utils.plot_data_to_numpy(
              lf0[0, 0, :].cpu().numpy(),
              pred_lf0[0, 0, :].detach().cpu().numpy(),
            ),
            'all/norm_lf0': utils.plot_data_to_numpy(
              lf0[0, 0, :].cpu().numpy(),
              norm_lf0[0, 0, :].detach().cpu().numpy(),
            ),
          })

        utils.summarize(
            writer=writer,
            global_step=global_step,
            images=image_dict,
            scalars=scalar_dict,
        )
      # 达到保存步数或者 stop 文件存在
      if global_step > 0 and global_step % hps.train.eval_interval == 0 or os.path.exists(os.path.join(hps.model_dir, 'stop.txt')):
        if os.path.exists(os.path.join(hps.model_dir, 'stop.txt')):
          logger.info('stop.txt found, stop training')
        evaluate(hps, net_g, eval_loader, writer_eval)
        save_checkpoints(net_g, optim_g, net_d, optim_d, hps, epoch)
        keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
        if keep_ckpts > 0:
          utils.clean_checkpoints(
            path_to_models=hps.model_dir,
            n_ckpts_to_keep=keep_ckpts,
            sort_by_time=True,
          )
        if os.path.exists(os.path.join(hps.model_dir, 'stop.txt')):
          logger.info('good bye!')
          os.remove(os.path.join(hps.model_dir, 'stop.txt'))
          os._exit(0)
    global_step += 1
    progress.advance(task)
  progress.remove_task(task)
  if rank == 0:
    global start_time
    now = time.time()
    duration = now - start_time  # 这里原本是 durtaion，让我看看是谁拼错了（
    logger.info(f'Epoch: {epoch} finished, cost {duration:.2f} s, {round(duration/len(train_loader),3)}s per batch')
    start_time = now


def evaluate(hps, generator, eval_loader, writer_eval):
  generator.eval()
  image_dict = {}
  audio_dict = {}
  with torch.no_grad():
    for batch_idx, items in enumerate(eval_loader):
      c, f0, spec, y, spk, _, uv, volume = items
      g = spk[:1].cuda(0)
      spec, y = spec[:1].cuda(0), y[:1].cuda(0)
      c = c[:1].cuda(0)
      f0 = f0[:1].cuda(0)
      uv = uv[:1].cuda(0)
      if volume is not None:
        volume = volume[:1].cuda(0)
      mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
      )
      y_hat, _ = generator.module.infer(c, f0, uv, g=g, vol=volume)

      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
      )

      audio_dict.update({f'gen/audio_{batch_idx}': y_hat[0], f'gt/audio_{batch_idx}': y[0]})
    image_dict.update({
      'gen/mel': utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
      'gt/mel': utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
    })
  utils.summarize(
    writer=writer_eval,
    global_step=global_step,
    images=image_dict,
    audios=audio_dict,
    audio_sampling_rate=hps.data.sampling_rate,
  )
  generator.train()


if __name__ == '__main__':
  main()
