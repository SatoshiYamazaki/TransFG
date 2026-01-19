import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from train import setup, valid
from utils.data_utils import get_loader, build_output_paths, canonical_dataset_name
from utils.dataset import write_labelmap, default_labels_for_dataset
from utils.dist_util import detect_device, write_env_stamp

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Eval run name (used under output/<name>)")
    parser.add_argument(
        "--dataset",
        choices=["flowers-102", "flower102", "CUB_200_2011", "car", "dog", "nabirds", "INat2017", "synthetic"],
        default="flowers-102",
        help="Dataset (contract enum: flowers-102|synthetic|CUB_200_2011|car|dog|nabirds|INat2017). Alias flower102 accepted.",
    )
    default_data_root = os.environ.get("DATA_ROOT", "./data")
    parser.add_argument("--data_root", type=str, default=default_data_root, help="Root directory containing the dataset folder (ignored for synthetic)")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "testing"], default="ViT-B_16")
    parser.add_argument("--pretrained_dir", type=str, default="./weights/ViT-B_16.npz", help="Path to ViT npz weights")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint (.bin) to load")
    parser.add_argument("--output_dir", type=str, default="./output", help="Base output directory for logs/artifacts")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--tiny_infer_subset", type=str, default="", help="Identifier/path for tiny inference subset metadata (e.g., flower102_tiny)")
    parser.add_argument("--tiny_train_subset", type=str, default="", help="Optional metadata field for tiny train subset")
    parser.add_argument("--prefer_mps", action="store_true", dest="prefer_mps", help="Prefer Apple MPS backend when available")
    parser.add_argument("--no-prefer-mps", action="store_false", dest="prefer_mps", help="Disable Apple MPS preference")
    parser.set_defaults(prefer_mps=True)
    parser.add_argument("--fiftyone_output", type=str, default=None, help="Path to write FiftyOne-compatible predictions JSONL (default: output/<name>/fiftyone/predictions.jsonl)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()
    args.dataset = canonical_dataset_name(args.dataset)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args.local_rank = -1
    paths = build_output_paths(args.output_dir, args.name)
    if args.fiftyone_output is None:
        args.fiftyone_output = str(paths["fiftyone_path"])
    data_root = Path(args.data_root)
    dataset_dir_matches = canonical_dataset_name(data_root.name) == args.dataset
    if args.dataset != "synthetic" and not dataset_dir_matches:
        data_root = data_root / args.dataset
    args.data_root = str(data_root)

    if args.dataset != "synthetic" and not data_root.exists():
        message = (
            f"Dataset directory not found: {data_root}. "
            "Provide --data_root (or DATA_ROOT) pointing to the folder that contains "
            f"{args.dataset}."
        )
        logger.error(message)
        raise FileNotFoundError(message)

    args.device = detect_device(prefer_mps=args.prefer_mps)
    args.n_gpu = torch.cuda.device_count() if args.device.type == "cuda" else (1 if args.device.type == "mps" else 0)
    args.nprocs = args.n_gpu if args.n_gpu else 1
    args.seed = getattr(args, "seed", 42)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        message = (
            f"Checkpoint not found at {checkpoint_path}. "
            "Pass --checkpoint pointing to a .bin file produced by train.py."
        )
        logger.error(message)
        raise FileNotFoundError(message)

    _, model = setup(args)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded checkpoint %s", args.checkpoint)

    stamp_path = paths["env_stamp_path"]
    write_env_stamp(args, stamp_path, args.device)
    write_labelmap(default_labels_for_dataset(args.dataset, args.num_classes), paths["labelmap_path"])
    paths["retention_path"].write_text(
        "\n".join([
            "Retention: 90 days minimum",
            f"Location: {paths['run_dir']}/",
            "Artifacts: TensorBoard logs, FiftyOne predictions, env_stamp.json",
            "Notes: Checkpoints are not committed; delete after TTL unless extended.",
        ])
    )

    # Data
    _, test_loader = get_loader(args)

    writer = SummaryWriter(log_dir=str(paths["tb_dir"]))
    with torch.no_grad():
        valid(args, model, writer, test_loader, global_step=0, fiftyone_path=Path(args.fiftyone_output))
    writer.close()
    logger.info("Eval run complete")

if __name__ == "__main__":
    main()
