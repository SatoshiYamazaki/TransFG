import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from train import setup, valid
from utils.data_utils import get_loader, build_output_paths
from utils.dist_util import detect_device, write_env_stamp

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of this eval run")
    parser.add_argument("--dataset", choices=["flower102", "CUB_200_2011", "car", "dog", "nabirds", "INat2017", "synthetic"], default="flower102")
    default_data_root = os.environ.get("DATA_ROOT", "./data")
    parser.add_argument("--data_root", type=str, default=default_data_root, help="Root directory for datasets")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "testing"], default="ViT-B_16")
    parser.add_argument("--pretrained_dir", type=str, default="./weights/ViT-B_16.npz", help="Path to ViT npz weights")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint (.bin) to load")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for logs/artifacts")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--tiny_infer_subset", type=str, default="", help="Path to tiny inference subset (optional metadata)")
    parser.add_argument("--tiny_train_subset", type=str, default="", help="Optional metadata field for tiny train subset")
    parser.add_argument("--prefer_mps", action="store_true", dest="prefer_mps", help="Prefer Apple MPS backend when available")
    parser.add_argument("--no-prefer-mps", action="store_false", dest="prefer_mps", help="Disable Apple MPS preference")
    parser.set_defaults(prefer_mps=True)
    parser.add_argument("--fiftyone_output", type=str, default=None, help="Path to write FiftyOne-compatible predictions JSONL")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args.local_rank = -1
    paths = build_output_paths(args.output_dir, args.name)
    if args.fiftyone_output is None:
        args.fiftyone_output = str(paths["fiftyone_path"])
    data_root = Path(args.data_root)
    if args.dataset != "synthetic" and data_root.name != args.dataset:
        data_root = data_root / args.dataset
    args.data_root = str(data_root)
    args.device = detect_device(prefer_mps=args.prefer_mps)
    args.n_gpu = torch.cuda.device_count() if args.device.type == "cuda" else (1 if args.device.type == "mps" else 0)
    args.nprocs = args.n_gpu if args.n_gpu else 1
    args.seed = getattr(args, "seed", 42)

    _, model = setup(args)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded checkpoint %s", args.checkpoint)

    stamp_path = paths["env_stamp_path"]
    write_env_stamp(args, stamp_path, args.device)

    # Data
    _, test_loader = get_loader(args)

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb", f"{args.name}"))
    with torch.no_grad():
        valid(args, model, writer, test_loader, global_step=0, fiftyone_path=Path(args.fiftyone_output))
    writer.close()
    logger.info("Eval run complete")

if __name__ == "__main__":
    main()
