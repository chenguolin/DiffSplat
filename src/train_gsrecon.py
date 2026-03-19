import warnings
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

from typing import *
from torch.nn.parallel import DistributedDataParallel
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from accelerate.data_loader import DataLoaderShard

import os
import argparse
import types
import logging
import math
import gc

from tqdm import tqdm
import wandb

import torch
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin
from lpips import LPIPS

from src.options import opt_dict
from src.data import GObjaverseParquetDataset, ParquetChunkDataSource, MultiEpochsChunkedDataLoader, yield_forever
from src.models import GSRecon, get_optimizer, get_lr_scheduler
import src.utils.util as util
import src.utils.vis_util as vis_util

from extensions.diffusers_diffsplat import MyEMAModel


def main():
    PROJECT_NAME = "GSRecon"

    parser = argparse.ArgumentParser(
        description="Train a multi-view 3DGS reconstruction Model"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="Path to the HDFS directory to save checkpoints"
    )
    parser.add_argument(
        "--wandb_token_path",
        type=str,
        default="wandb/token",
        help="Path to the WandB login token"
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=None,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        help="Use offline WandB for experiment tracking"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training"
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=5,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the data loader"
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model for training"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed"
    )

    parser.add_argument(
        "--load_pretrained_model",
        type=str,
        default=None,
        help="Tag of the model pretrained in this project"
    )
    parser.add_argument(
        "--load_pretrained_model_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained model checkpoint"
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()

    # Parse the config file
    configs = util.get_configs(args.config_file, extras)  # change yaml configs by `extras`

    # Parse the option dict
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)
    opt.__post_init__()

    # Create an experiment directory using the `tag`
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.hdfs_dir is not None:
        args.project_hdfs_dir = args.hdfs_dir
        args.hdfs_dir = os.path.join(args.hdfs_dir, args.tag)
        os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir}")

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(os.path.join(exp_dir, "log.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True  # propagate to the root logger (console)

    # Set DeepSpeed config
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
            zero_stage=int(args.zero_stage),
            offload_optimizer_device="cpu",  # hard-coded here, TODO: make it configurable
        )
    else:
        deepspeed_plugin = None

    # Initialize the accelerator
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,  # batch size per GPU
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
        deepspeed_plugin=deepspeed_plugin,
    )
    logger.info(f"Accelerator state:\n{accelerator.state}\n")

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Prepare dataset
    if accelerator.is_local_main_process:
        if not os.path.exists("/tmp/test_dataset"):
            os.system(opt.dataset_setup_script)
    accelerator.wait_for_everyone()  # other processes wait for the main process

    # Load the training and validation dataset
    assert opt.file_dir_train is not None and opt.file_name_train is not None and \
        opt.file_dir_test is not None and opt.file_name_test is not None

    train_dataset = GObjaverseParquetDataset(
        data_source=ParquetChunkDataSource(opt.file_dir_train, opt.file_name_train),
        shuffle=True,
        shuffle_buffer_size=-1,  # `-1`: not shuffle actually
        chunks_queue_max_size=1,  # number of preloading chunks
        # GObjaverse
        opt=opt,
        training=True,
    )
    val_dataset = GObjaverseParquetDataset(
        data_source=ParquetChunkDataSource(opt.file_dir_test, opt.file_name_test),
        shuffle=True,  # shuffle for various visualization
        shuffle_buffer_size=-1,  # `-1`: not shuffle actually
        chunks_queue_max_size=1,  # number of preloading chunks
        # GObjaverse
        opt=opt,
        training=False,
    )
    train_loader = MultiEpochsChunkedDataLoader(
        train_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )
    val_loader = MultiEpochsChunkedDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )

    logger.info(f"Load [{len(train_dataset)}] training samples and [{len(val_dataset)}] validation samples\n")

    # Compute the effective batch size and scale learning rate
    total_batch_size = configs["train"]["batch_size_per_gpu"] * \
        accelerator.num_processes * args.gradient_accumulation_steps
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= (total_batch_size / 256)
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]

    # LPIPS loss
    if accelerator.is_main_process:
        _ = LPIPS(net="vgg")
        del _
    accelerator.wait_for_everyone()  # wait for pretrained backbone weights to be downloaded
    lpips_loss = LPIPS(net="vgg").to(accelerator.device)
    lpips_loss = lpips_loss.requires_grad_(False)
    lpips_loss.eval()

    # For DeepSpeed bug: model inputs could be `torch.nn.Module` (e.g., `lpips_loss`)
    def is_floating_point(self):
        return True
    lpips_loss.is_floating_point = types.MethodType(is_floating_point, lpips_loss)

    # Initialize the model, optimizer and lr scheduler
    model = GSRecon(opt)
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = get_optimizer(params=params_to_optimize, **configs["optimizer"])

    configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
        len(train_loader) // accelerator.num_processes / args.gradient_accumulation_steps)  # only account updated steps
    configs["lr_scheduler"]["total_steps"] *= accelerator.num_processes  # for lr scheduler setting
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] *= accelerator.num_processes  # for lr scheduler setting
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"]["total_steps"] //= accelerator.num_processes  # reset for multi-gpu
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] //= accelerator.num_processes  # reset for multi-gpu

    # Load a pretrained model
    if args.load_pretrained_model is not None:
        logger.info(f"Load GSRecon checkpoint from [{args.load_pretrained_model}] iteration [{args.load_pretrained_model_ckpt:06d}]\n")
        model = util.load_ckpt(
            os.path.join(args.output_dir, args.load_pretrained_model, "checkpoints"),
            args.load_pretrained_model_ckpt,
            None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_model),
            model, accelerator
        )

    # Initialize the EMA model to save moving average states
    if args.use_ema:
        logger.info("Use exponential moving average (EMA) for model parameters\n")
        ema_states = MyEMAModel(
            model.parameters(),
            **configs["train"]["ema_kwargs"]
        )
        ema_states.to(accelerator.device)

    # Prepare everything with `accelerator`
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader, val_loader
    )
    # Set classes explicitly for everything
    model: DistributedDataParallel
    optimizer: AcceleratedOptimizer
    lr_scheduler: AcceleratedScheduler
    train_loader: DataLoaderShard
    val_loader: DataLoaderShard

    # Cast input dataset to the appropriate dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Training configs after distribution and accumulation setup
    updated_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updated_steps = configs["lr_scheduler"]["total_steps"]
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps
    assert configs["train"]["epochs"] * updated_steps_per_epoch == total_updated_steps
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: [{len(val_loader)}]\n")

    # (Optional) Load checkpoint
    global_update_step = 0
    if args.resume_from_iter is not None:
        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        # Download from HDFS
        if not os.path.exists(os.path.join(ckpt_dir, f'{args.resume_from_iter:06d}')):
            args.resume_from_iter = util.load_ckpt(
                ckpt_dir,
                args.resume_from_iter,
                args.hdfs_dir,
                None,  # `None`: not load model ckpt here
                accelerator,  # manage the process states
            )
        # Load everything
        accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"))  # torch < 2.4.0 here for `weights_only=False`
        if args.use_ema:
            ema_states.load_state_dict(torch.load(
                os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}", "ema_states.pth"),
                map_location=accelerator.device,
            ))
        global_update_step = int(args.resume_from_iter)

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = util.save_experiment_params(args, configs, opt, exp_dir)
        util.save_model_architecture(accelerator.unwrap_model(model), exp_dir)

    # WandB logger
    if accelerator.is_main_process:
        if args.offline_wandb:
            os.environ["WANDB_MODE"] = "offline"
        with open(args.wandb_token_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()
        wandb.init(
            project=PROJECT_NAME, name=args.tag,
            config=exp_params, dir=exp_dir,
            resume=True
        )
        # Wandb artifact for logging experiment information
        arti_exp_info = wandb.Artifact(args.tag, type="exp_info")
        arti_exp_info.add_file(os.path.join(exp_dir, "params.yaml"))
        arti_exp_info.add_file(os.path.join(exp_dir, "model.txt"))
        arti_exp_info.add_file(os.path.join(exp_dir, "log.txt"))  # only save the log before training
        wandb.log_artifact(arti_exp_info)

    # Start training
    logger.logger.propagate = False  # not propagate to the root logger (console)
    progress_bar = tqdm(
        range(total_updated_steps),
        initial=global_update_step,
        desc="Training",
        ncols=125,
        disable=not accelerator.is_main_process
    )
    for batch in yield_forever(train_loader):

        if global_update_step == args.max_train_steps:
            progress_bar.close()
            logger.logger.propagate = True  # propagate to the root logger (console)
            if accelerator.is_main_process:
                wandb.finish()
            logger.info("Training finished!\n")
            return

        model.train()

        with accelerator.accumulate(model):
            outputs = model(batch, lpips_loss, step=global_update_step+1, dtype=weight_dtype)  # `step` starts from 1

            psnr = outputs["psnr"]
            ssim = outputs["ssim"]
            lpips = outputs["lpips"]
            loss = outputs["loss"]
            if "coord_mse" in outputs:
                coord_mse = outputs["coord_mse"]
            else:
                coord_mse = None
            if "normal_cosim" in outputs:
                normal_cosim = outputs["normal_cosim"]
            else:
                normal_cosim = None

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            # Gather the losses across all processes for logging (if we use distributed training)
            psnr = accelerator.gather(psnr.detach()).mean()
            ssim = accelerator.gather(ssim.detach()).mean()
            lpips = accelerator.gather(lpips.detach()).mean()
            loss = accelerator.gather(loss.detach()).mean()
            if coord_mse is not None:
                coord_mse = accelerator.gather(coord_mse.detach()).mean()
            if normal_cosim is not None:
                normal_cosim = accelerator.gather(normal_cosim.detach()).mean()

            logs = {
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            if args.use_ema:
                ema_states.step(model.parameters())
                logs.update({"ema": ema_states.cur_decay_value})

            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            global_update_step += 1

            logger.info(
                f"[{global_update_step:06d} / {total_updated_steps:06d}] " +
                f"psnr: {logs['psnr']:.4f}, ssim: {logs['ssim']:.4f}, lpips: {logs['lpips']:.4f}, " +
                f"loss: {logs['loss']:.4f}, lr: {logs['lr']:.2e}" +
                f", ema: {logs['ema']:.4f}" if args.use_ema else ""
            )

            # Log the training progress
            if global_update_step % configs["train"]["log_freq"] == 0 or global_update_step == 1 \
                or global_update_step % updated_steps_per_epoch == 0:  # last step of an epoch
                if accelerator.is_main_process:
                    wandb.log({
                        "training/psnr": psnr.item(),
                        "training/ssim": ssim.item(),
                        "training/lpips": lpips.item(),
                        "training/loss": loss.item(),
                        "training/lr": lr_scheduler.get_last_lr()[0]
                    }, step=global_update_step)
                    if coord_mse is not None:
                        wandb.log({
                            "training/coord_mse": coord_mse.item()
                        }, step=global_update_step)
                    if normal_cosim is not None:
                        wandb.log({
                            "training/normal_cosim": normal_cosim.item()
                        }, step=global_update_step)
                    if args.use_ema:
                        wandb.log({
                            "training/ema": ema_states.cur_decay_value
                        }, step=global_update_step)

            # Save checkpoint
            if (global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                or global_update_step == total_updated_steps):  # 3. last step of an epoch

                gc.collect()
                if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                elif accelerator.is_main_process:
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                accelerator.wait_for_everyone()  # ensure all processes have finished saving
                if accelerator.is_main_process:
                    if args.use_ema:
                        torch.save(ema_states.state_dict(),
                            os.path.join(ckpt_dir, f"{global_update_step:06d}", "ema_states.pth"))
                    if args.hdfs_dir is not None:
                        util.save_ckpt(ckpt_dir, global_update_step, args.hdfs_dir)
                gc.collect()

            # Evaluate on the validation set
            if (global_update_step == 1
                or (global_update_step % configs["train"]["early_eval_freq"] == 0 and
                global_update_step < configs["train"]["early_eval"])  # 1. more frequently at the beginning
                or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                or global_update_step == total_updated_steps):  # 4. last step of an epoch

                torch.cuda.empty_cache()
                gc.collect()

                # Use EMA parameters for evaluation
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference
                    ema_states.store(model.parameters())
                    ema_states.copy_to(model.parameters())

                with torch.no_grad():
                    with torch.autocast("cuda", torch.bfloat16):
                        model.eval()

                        all_val_metrics, val_steps = {}, 0
                        val_progress_bar = tqdm(
                            range(len(val_loader)) if args.max_val_steps is None \
                                else range(args.max_val_steps),
                            desc="Validation",
                            ncols=125,
                            disable=not accelerator.is_main_process
                        )
                        for val_batch in val_loader:
                            val_outputs = model(val_batch, lpips_loss, step=global_update_step, dtype=weight_dtype)

                            val_psnr = val_outputs["psnr"]
                            val_ssim = val_outputs["ssim"]
                            val_lpips = val_outputs["lpips"]
                            val_loss = val_outputs["loss"]
                            if "coord_mse" in val_outputs:
                                val_coord_mse = val_outputs["coord_mse"]
                            else:
                                val_coord_mse = None
                            if "normal_cosim" in val_outputs:
                                val_normal_cosim = val_outputs["normal_cosim"]
                            else:
                                val_normal_cosim = None

                            val_psnr = accelerator.gather_for_metrics(val_psnr).mean()
                            val_ssim = accelerator.gather_for_metrics(val_ssim).mean()
                            val_lpips = accelerator.gather_for_metrics(val_lpips).mean()
                            val_loss = accelerator.gather_for_metrics(val_loss).mean()
                            if val_coord_mse is not None:
                                val_coord_mse = accelerator.gather_for_metrics(val_coord_mse).mean()
                            if val_normal_cosim is not None:
                                val_normal_cosim = accelerator.gather_for_metrics(val_normal_cosim).mean()

                            val_logs = {
                                "psnr": val_psnr.item(),
                                "ssim": val_ssim.item(),
                                "lpips": val_lpips.item(),
                                "loss": val_loss.item()
                            }
                            val_progress_bar.set_postfix(**val_logs)
                            val_progress_bar.update(1)
                            val_steps += 1

                            all_val_metrics.setdefault("psnr", []).append(val_psnr)
                            all_val_metrics.setdefault("ssim", []).append(val_ssim)
                            all_val_metrics.setdefault("lpips", []).append(val_lpips)
                            all_val_metrics.setdefault("loss", []).append(val_loss)
                            if val_coord_mse is not None:
                                all_val_metrics.setdefault("coord_mse", []).append(val_coord_mse)
                            if val_normal_cosim is not None:
                                all_val_metrics.setdefault("normal_cosim", []).append(val_normal_cosim)

                            if args.max_val_steps is not None and val_steps == args.max_val_steps:
                                break

                val_progress_bar.close()

                if args.use_ema:
                    # Switch back to the original model parameters
                    ema_states.restore(model.parameters())

                for k, v in all_val_metrics.items():
                    all_val_metrics[k] = torch.tensor(v).mean()

                logger.info(
                    f"Eval [{global_update_step:06d} / {total_updated_steps:06d}] " +
                    f"psnr: {all_val_metrics['psnr'].item():.4f}, " +
                    f"ssim: {all_val_metrics['ssim'].item():.4f}, " +
                    f"lpips: {all_val_metrics['lpips'].item():.4f}, " +
                    f"loss: {all_val_metrics['loss'].item():.4f}\n"
                )

                if accelerator.is_main_process:
                    wandb.log({
                        "validation/psnr": all_val_metrics["psnr"].item(),
                        "validation/ssim": all_val_metrics["ssim"].item(),
                        "validation/lpips": all_val_metrics["lpips"].item(),
                        "validation/loss": all_val_metrics["loss"].item()
                    }, step=global_update_step)
                    if "coord_mse" in all_val_metrics:
                        wandb.log({
                            "validation/coord_mse": all_val_metrics["coord_mse"].item()
                        }, step=global_update_step)
                    if "normal_cosim" in all_val_metrics:
                        wandb.log({
                            "validation/normal_cosim": all_val_metrics["normal_cosim"].item()
                        }, step=global_update_step)

                    # Visualize rendering
                    wandb.log({
                        "images/training": vis_util.wandb_mvimage_log(outputs)
                    }, step=global_update_step)
                    wandb.log({
                        "images/validation": vis_util.wandb_mvimage_log(val_outputs)
                    }, step=global_update_step)

                torch.cuda.empty_cache()
                gc.collect()


if __name__ == "__main__":
    main()
