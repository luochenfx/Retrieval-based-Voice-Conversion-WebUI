import os
import sys
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

import datetime

from infer.lib.train import utils

hps = utils.get_hparams()

n_gpus = len(hps.gpus.split("-"))
from random import randint, shuffle

import torch

if hasattr(torch.version, "hip") and torch.version.hip is not None:
    # ROCm 环境
    os.environ["ROCR_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
    # HSA_VISIBLE_DEVICES（旧版本兼容）
    # os.environ["HSA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
else:
    # CUDA 环境
    os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")

# PyTorch >= 2.4.0
try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init
        from infer.modules.ipex.gradscaler import gradscaler_init
        from torch.amp import autocast

        GradScaler = gradscaler_init()
        ipex_init()
    else:
        from torch.amp import GradScaler, autocast
except Exception:
    from torch.amp import GradScaler, autocast

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.lib.infer_pack import commons
from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

if hps.version == "v1":
    from infer.lib.infer_pack.models import MultiPeriodDiscriminator
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.train.process_ckpt import savee

global_step = 0
device = "cuda"
show_postfix = os.environ.get('RVC_SHOW_POSTFIX', '0').lower() in ('1', 'true', 'yes')


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    n_gpus = torch.cuda.device_count()

    if torch.cuda.is_available() == False and torch.backends.mps.is_available() == True:
        n_gpus = 1
    if n_gpus < 1:
        # patch to unblock people without gpus. there is probably a better way.
        print("="*60)
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    logger = utils.get_logger(hps.model_dir)
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(rank, n_gpus, hps, logger: logging.Logger):
    global global_step
    if rank == 0:
        # logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        logger.info(f"log_dir: {hps.model_dir}")
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.to(device)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.to(device)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        pass
    elif hasattr(torch.version, "hip") and torch.version.hip is not None:
        logger.info(f"Detected ROCm/HIP backend (version: {torch.version.hip})")
    elif torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            if hasattr(net_g, "module"):
                logger.info(
                    net_g.module.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu")["model"]
                    )
                )  ##测试不加载优化器
            else:
                logger.info(
                    net_g.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu")["model"]
                    )
                )  ##测试不加载优化器
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )
            else:
                logger.info(
                    net_d.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # ── 数据迭代器准备 ────────────────────────────────────────────────────────
    if hps.if_cache_data_in_gpu:
        # 每个 epoch 都强制重建缓存（安全）
        cache.clear()
        logger.debug(f"Epoch {epoch}: Rebuilding GPU cache...")

        for batch_idx, batch in enumerate(train_loader):
            # Unpack
            if hps.if_f0 == 1:
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = batch
            else:
                phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = batch

            # Load on CUDA
            if torch.cuda.is_available():
                phone = phone.to(device, non_blocking=True)
                phone_lengths = phone_lengths.to(device, non_blocking=True)
                if hps.if_f0 == 1:
                    pitch = pitch.to(device, non_blocking=True)
                    pitchf = pitchf.to(device, non_blocking=True)
                sid = sid.to(device, non_blocking=True)
                spec = spec.to(device, non_blocking=True)
                spec_lengths = spec_lengths.to(device, non_blocking=True)
                wave = wave.to(device, non_blocking=True)
                # wave_lengths = wave_lengths.to(device, non_blocking=True)  # 如果需要

            # 存入缓存（不带 batch_idx，避免无用信息）
            if hps.if_f0 == 1:
                cache.append((phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid))
            else:
                cache.append((phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid))

        logger.debug(f"GPU cache rebuilt: {len(cache)} batches")

        # 打乱顺序
        shuffle(cache)

        # 用 enumerate 包装，让 batch_idx 从 0 开始
        data_iterator = enumerate(cache)
        iterator_length = len(cache)
    else:
        # 普通 loader 模式
        data_iterator = enumerate(train_loader)
        iterator_length = len(train_loader)

    if iterator_length > 0:
        hps.train.log_interval = max(1, iterator_length // 5)
    else:
        hps.train.log_interval = 1
    # logger.info(f"自适应 log_interval: {hps.train.log_interval} (基于 {iterator_length} batch/epoch)")
    # ── 训练循环 ───────────────────────────────────────────────────────────────
    epoch_recorder = EpochRecorder()
    pbar = tqdm(
        data_iterator,
        total=iterator_length,
        desc=f"Epoch {epoch}",
        dynamic_ncols=True,
        leave=True,
        unit="batch"
    )

    for batch_idx, batch_data in pbar:
        # 统一解包（缓存和 loader 都一样格式）
        if hps.if_f0 == 1:
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = batch_data
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = batch_data

        # Calculate ─────────────────────────────────────────────────────────────
        with autocast(device, enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
                )
            else:
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                    phone, phone_lengths, spec, spec_lengths, sid
                )

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )

            with autocast(device, enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

            if hps.train.fp16_run:
                y_hat_mel = y_hat_mel.half()

            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(device, enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )

        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(device, enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(device, enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        # 日志与可视化
        if rank == 0 and global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            display_mel = min(loss_mel.item(), 75)
            display_kl = min(loss_kl.item(), 9)

            scalar_dict = {
                "loss/g/total": loss_gen_all.item(),
                "loss/d/total": loss_disc.item(),
                "learning_rate": lr,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
                "loss/g/fm": loss_fm.item(),
                "loss/g/mel": display_mel,
                "loss/g/kl": display_kl,
            }
            scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
            scalar_dict.update({f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)})
            scalar_dict.update({f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)})

            image_dict = {}
            if global_step % (hps.train.log_interval * 10) == 0:
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                }

            utils.summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict,
            )
        # tqdm postfix 只在 log_interval 更新
        if show_postfix and global_step % hps.train.log_interval == 0:
            postfix_dict = {
                "g": f"{loss_gen_all.item():.3f}",
                "d": f"{loss_disc.item():.3f}",
                "mel": f"{loss_mel.item():.3f}",
                "kl": f"{loss_kl.item():.3f}",
            }
            pbar.set_postfix(postfix_dict)

        global_step += 1

    # ── 保存检查点（保持原样） ─────────────────────────────────────────────────
    if epoch % hps.save_every_epoch == 0 and rank == 0:
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, f"G_{global_step}.pth")
            )
            utils.save_checkpoint(
                net_d, optim_d, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, f"D_{global_step}.pth")
            )
        else:
            utils.save_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, "G_2333333.pth")
            )
            utils.save_checkpoint(
                net_d, optim_d, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, "D_2333333.pth")
            )

        if rank == 0 and hps.save_every_weights == "1":
            ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s" % (
                    hps.name, epoch,
                    savee(ckpt, hps.sample_rate, hps.if_f0, f"{hps.name}_e{epoch}_s{global_step}",
                          epoch, hps.version, hps)
                )
            )

    if rank == 0:
        logger.info(f"====> Epoch: {epoch} {epoch_recorder.record()}")

    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")
        ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
        logger.info("saving final ckpt:%s" % savee(
            ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
        ))
        sleep(1)
        os._exit(2333333)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
