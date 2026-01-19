# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, build_output_paths
from utils.dataset import write_labelmap, default_labels_for_dataset
from utils.dist_util import get_world_size, detect_device, set_seed_all, write_env_stamp

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    preds = preds.view(-1)
    labels = labels.view(-1)
    return (preds == labels).float().mean()

def reduce_mean(tensor, nprocs):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= max(1, nprocs)
    return rt

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_checkpoint.bin")
    checkpoint = {'model': model_to_save.state_dict()}
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    elif args.dataset == "synthetic":
        num_classes = 4
    elif args.dataset == "flowers-102":
        num_classes = 102
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    args.num_classes = num_classes
    
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,                                                   smoothing_value=args.smoothing_value)

    if args.pretrained_dir and os.path.exists(args.pretrained_dir):
        model.load_from(np.load(args.pretrained_dir))
    else:
        logger.warning("Pretrained weights not found at %s; initializing randomly", args.pretrained_dir)
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model, map_location=args.device).get('model', None)
        if pretrained_model is not None:
            model.load_state_dict(pretrained_model)
        else:
            logger.warning("Checkpoint %s missing 'model' key; skipping state dict load", args.pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def valid(args, model, writer, test_loader, global_step, fiftyone_path: Optional[Path] = None):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    sample_records = []
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if fiftyone_path is not None:
            preds_cpu = preds.detach().cpu().tolist()
            labels_cpu = y.detach().cpu().tolist()
            base_index = step * args.eval_batch_size
            for local_idx, (p, l) in enumerate(zip(preds_cpu, labels_cpu)):
                sample_records.append({
                    "sample_id": int(base_index + local_idx),
                    "prediction": int(p),
                    "label": int(l)
                })

        all_preds.append(preds.detach().cpu())
        all_label.append(y.detach().cpu())
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds_t = torch.cat(all_preds)
    all_label_t = torch.cat(all_label)
    accuracy = simple_accuracy(all_preds_t, all_label_t).to(args.device)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    val_accuracy = reduce_mean(accuracy, args.nprocs)
    val_accuracy = float(val_accuracy.detach().cpu().item())

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)
    if args.local_rank in [-1, 0]:
        writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=global_step)

    if fiftyone_path is not None and args.local_rank in [-1, 0]:
        fiftyone_path.parent.mkdir(parents=True, exist_ok=True)
        with fiftyone_path.open("w") as fout:
            for rec in sample_records:
                fout.write(json.dumps(rec) + "\n")
        logger.info("Wrote FiftyOne-compatible predictions to %s", fiftyone_path)
        
    return val_accuracy

def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    paths = build_output_paths(args.output_dir, args.name)
    writer = SummaryWriter(log_dir=str(paths["tb_dir"]))

    num_classes = getattr(args, "num_classes", None)
    if num_classes is None:
        num_classes = getattr(getattr(model, "classifier", None), "out_features", None)
    if num_classes is None:
        num_classes = 102 if args.dataset == "flowers-102" else 1
    args.num_classes = num_classes

    if args.local_rank in [-1, 0]:
        write_labelmap(default_labels_for_dataset(args.dataset, args.num_classes), paths["labelmap_path"])
        write_env_stamp(args, paths["env_stamp_path"], args.device)
        paths["retention_path"].write_text(
            "\n".join([
                "Retention: 90 days minimum",
                f"Location: {paths['run_dir']}/",
                "Artifacts: TensorBoard logs, FiftyOne predictions, env_stamp.json",
                "Notes: Checkpoints are not committed; delete after TTL unless extended.",
            ])
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per device = %d", args.train_batch_size)
    logger.info("  Total train batch size (parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed_all(args.seed)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    start_time = time.time()
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            loss, logits = model(x, y)
            loss = loss.mean()

            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.detach().cpu())
            all_label.append(y.detach().cpu())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0:
                    with torch.no_grad():
                        accuracy = valid(
                            args,
                            model,
                            writer,
                            test_loader,
                            global_step,
                            fiftyone_path=paths["fiftyone_path"] if args.local_rank in [-1, 0] else None,
                        )
                    if args.local_rank in [-1, 0]:
                        if best_acc < accuracy:
                            save_model(args, model)
                            best_acc = accuracy
                        logger.info("best accuracy so far: %f" % best_acc)
                    model.train()

                if global_step % t_total == 0:
                    break
        all_preds_t = torch.cat(all_preds)
        all_label_t = torch.cat(all_label)
        accuracy = simple_accuracy(all_preds_t, all_label_t).to(args.device)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        train_accuracy = reduce_mean(accuracy, args.nprocs)
        train_accuracy = float(train_accuracy.detach().cpu().item())
        logger.info("train accuracy so far: %f" % train_accuracy)
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0] and args.fiftyone_output:
        with torch.no_grad():
            valid(
                args,
                model,
                writer,
                test_loader,
                global_step,
                fiftyone_path=Path(args.fiftyone_output),
            )

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["flowers-102", "CUB_200_2011", "car", "dog", "nabirds", "INat2017", "synthetic"], default="flowers-102", help="Which dataset.")
    default_data_root = os.environ.get("DATA_ROOT", "./data")
    parser.add_argument("--data_root", type=str, default=default_data_root, help="Root directory containing datasets; synthetic ignores this path.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "testing"], default="ViT-B_16", help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="./weights/ViT-B_16.npz", help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Optional fine-tuned checkpoint to load (bin file).")
    parser.add_argument("--output_dir", default="./output", type=str, help="The output directory where checkpoints/logs will be written.")
    parser.add_argument("--img_size", default=448, type=int, help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int, help="Run prediction on validation set every so many steps. Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=3e-2, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine", help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--smoothing_value", type=float, default=0.0, help="Label smoothing value")
    parser.add_argument("--split", type=str, default="non-overlap", help="Split method")
    parser.add_argument("--slide_step", type=int, default=12, help="Slide step for overlap split")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (set 0 for CPU/MPS debugging)")
    parser.add_argument("--prefer_mps", action="store_true", dest="prefer_mps", help="Prefer Apple MPS backend when available")
    parser.add_argument("--no-prefer-mps", action="store_false", dest="prefer_mps", help="Disable Apple MPS preference")
    parser.set_defaults(prefer_mps=True)
    parser.add_argument("--eval_only", action="store_true", help="Run validation only (no training) using the provided checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned checkpoint for eval_only or warm start")
    parser.add_argument("--fiftyone_output", type=str, default=None, help="Path to write FiftyOne-compatible JSONL predictions during eval")
    parser.add_argument("--tiny_train_subset", type=str, default="", help="Path to tiny train subset (for metadata/env stamp)")
    parser.add_argument("--tiny_infer_subset", type=str, default="", help="Path to tiny infer subset (for metadata/env stamp)")

    args = parser.parse_args()

    paths = build_output_paths(args.output_dir, args.name)
    if args.fiftyone_output is None:
        args.fiftyone_output = str(paths["fiftyone_path"])

    data_root = Path(args.data_root)
    if args.dataset != "synthetic" and data_root.name != args.dataset:
        data_root = data_root / args.dataset
    args.data_root = str(data_root)

    if args.local_rank == -1:
        device = detect_device(prefer_mps=args.prefer_mps)
        args.n_gpu = torch.cuda.device_count() if device.type == "cuda" else (1 if device.type == "mps" else 0)
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = args.n_gpu if args.n_gpu else 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))
    logger.info("Training/evaluation parameters %s", args)

    # Set seed
    set_seed_all(args.seed)

    # Model Setup
    args, model = setup(args)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict)
        logger.info("Loaded fine-tuned checkpoint from %s", args.checkpoint)

    stamp_path = paths["env_stamp_path"]
    write_env_stamp(args, stamp_path, args.device)

    # Training only
    train(args, model)

if __name__ == "__main__":
    main()
