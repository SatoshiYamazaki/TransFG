import logging
import os
import random
from functools import partial
from pathlib import Path
from typing import Tuple, List

import torch
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, Dataset, Subset
from torchvision import transforms

from .autoaugment import AutoAugImageNetPolicy
from .dataset import CUB, CarsDataset, NABirds, dogs, INat2017, Flowers102
from .dist_util import set_seed_all

logger = logging.getLogger(__name__)


def canonical_dataset_name(name: str) -> str:
    """Normalize dataset identifiers to match CLI/openapi contract."""
    aliases = {
        "flower102": "flowers-102",
        "flowers102": "flowers-102",
    }
    return aliases.get(name, name)


class SyntheticDataset(Dataset):
    """Tiny synthetic dataset for smoke tests on CPU/MPS."""

    def __init__(self, length: int = 16, num_classes: int = 3, image_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.length = length
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h, w = self.image_size
        torch.manual_seed(idx)
        image = torch.rand(3, h, w)
        label = torch.randint(low=0, high=self.num_classes, size=(1,)).item()
        return image, label


def worker_init_fn(worker_id: int, base_seed: int) -> None:
    """Top-level worker init to keep picklable under spawn context."""
    set_seed_all(base_seed + worker_id)


def maybe_limit_subset(dataset: Dataset, max_items: int) -> Dataset:
    if max_items is None or max_items <= 0:
        return dataset
    return Subset(dataset, list(range(min(len(dataset), max_items))))


def build_output_paths(output_dir: str, run_name: str) -> dict:
    base = Path(output_dir)
    run_dir = base / run_name
    return {
        "run_dir": run_dir,
        "tb_dir": base / "tb" / run_name,
        "fiftyone_path": run_dir / "fiftyone" / "predictions.jsonl",
        "labelmap_path": run_dir / "labelmap.json",
        "checkpoints_dir": run_dir / "checkpoints",
        "env_stamp_path": run_dir / "env_stamp.json",
        "retention_path": run_dir / "RETENTION.txt",
    }


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.dataset = canonical_dataset_name(getattr(args, "dataset", ""))

    num_workers = getattr(args, "num_workers", 2)

    base_seed = getattr(args, "seed", 42)
    worker_init = partial(worker_init_fn, base_seed=base_seed)

    if args.dataset == 'CUB_200_2011':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform = test_transform)
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root,'devkit/cars_train_annos.mat'),
                            os.path.join(args.data_root,'cars_train'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
        testset = CarsDataset(os.path.join(args.data_root,'cars_test_annos_withlabels.mat'),
                            os.path.join(args.data_root,'cars_test'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
    elif args.dataset == 'dog':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                                train=True,
                                cropped=False,
                                transform=train_transform,
                                download=False
                                )
        testset = dogs(root=args.data_root,
                                train=False,
                                cropped=False,
                                transform=test_transform,
                                download=False
                                )
    elif args.dataset == 'nabirds':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                        transforms.RandomCrop((448, 448)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                        transforms.CenterCrop((448, 448)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.RandomCrop((304, 304)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.CenterCrop((304, 304)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)
    elif args.dataset == 'flowers-102':
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            norm,
        ])
        trainset = Flowers102(root=args.data_root, split="train", download=False, transform=train_transform)
        testset = Flowers102(root=args.data_root, split="val", download=False, transform=test_transform)

        if getattr(args, "tiny_train_subset", "") == "flower102_tiny":
            trainset = maybe_limit_subset(trainset, args.train_batch_size * 2)
        if getattr(args, "tiny_infer_subset", "") == "flower102_tiny":
            testset = maybe_limit_subset(testset, args.eval_batch_size * 2)
    elif args.dataset == 'synthetic':
        trainset = SyntheticDataset(length=16, num_classes=4, image_size=(args.img_size, args.img_size))
        testset = SyntheticDataset(length=8, num_classes=4, image_size=(args.img_size, args.img_size))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=num_workers,
                              worker_init_fn=worker_init,
                              drop_last=True,
                              pin_memory=False)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=num_workers,
                             worker_init_fn=worker_init,
                             pin_memory=False) if testset is not None else None

    return train_loader, test_loader
