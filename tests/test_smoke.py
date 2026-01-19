import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import train
from models.modeling import CONFIGS, VisionTransformer
from utils.data_utils import SyntheticDataset, build_output_paths
from utils.dataset import write_labelmap, default_labels_for_dataset
from utils.dist_util import write_env_stamp


@pytest.mark.smoke_train
def test_smoke_train_flower102_tiny(tmp_path: Path, monkeypatch):
    """Tiny training smoke: forward/backward + predictions export using flower102 args."""

    # Build deterministic loaders using synthetic data but flower102 args
    train_ds = SyntheticDataset(length=4, num_classes=3, image_size=(32, 32))
    eval_ds = SyntheticDataset(length=4, num_classes=3, image_size=(32, 32))
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False)

    def fake_get_loader(args):
        return train_loader, eval_loader

    monkeypatch.setattr("train.get_loader", fake_get_loader)

    class DummyTrainModel(torch.nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            input_dim = 3 * 32 * 32
            self.classifier = torch.nn.Linear(input_dim, num_classes)

        def forward(self, x, labels=None):
            batch_size = x.shape[0]
            logits = self.classifier(x.view(batch_size, -1))
            if labels is None:
                return logits
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return loss, logits

    args = types.SimpleNamespace(
        name="flower_smoke",
        dataset="flower102",
        data_root=str(tmp_path),
        model_type="testing",
        img_size=32,
        train_batch_size=2,
        eval_batch_size=2,
        eval_every=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_steps=2,
        decay_type="linear",
        warmup_steps=0,
        max_grad_norm=1.0,
        local_rank=-1,
        seed=42,
        gradient_accumulation_steps=1,
        smoothing_value=0.0,
        split="non-overlap",
        slide_step=12,
        num_workers=0,
        prefer_mps=False,
        eval_only=False,
        checkpoint=None,
        nprocs=1,
        device=torch.device("cpu"),
        pretrained_dir="",
        pretrained_model=None,
    )

    paths = build_output_paths(str(tmp_path), args.name)
    args.output_dir = str(tmp_path)
    args.fiftyone_output = str(paths["fiftyone_path"])

    # Model setup
    model = DummyTrainModel(num_classes=3).to(args.device)
    writer = SummaryWriter(log_dir=paths["tb_dir"])

    # Run minimal train loop
    train.train(args, model)
    writer.close()

    preds_path = paths["fiftyone_path"]
    assert preds_path.exists()
    lines = preds_path.read_text().strip().splitlines()
    assert len(lines) == 4


@pytest.mark.smoke_eval
def test_valid_writes_preds_and_labelmap(tmp_path: Path):
    """Contract: predictions rows match two eval batches and labelmap exists."""

    class DummyClassifier(torch.nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.num_classes = num_classes

        def forward(self, x):
            batch_size = x.shape[0]
            return torch.zeros((batch_size, self.num_classes), device=x.device)

    model = DummyClassifier(num_classes=3).to(torch.device("cpu"))

    dataset = SyntheticDataset(length=4, num_classes=3, image_size=(32, 32))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    args = types.SimpleNamespace(
        device=torch.device("cpu"),
        eval_batch_size=2,
        local_rank=-1,
        nprocs=1,
        tiny_train_subset="flower102_tiny",
        tiny_infer_subset="flower102_tiny",
        name="flower_eval",
        dataset="flower102",
        output_dir=str(tmp_path),
        seed=42,
    )

    paths = build_output_paths(str(tmp_path), args.name)
    label_info = write_labelmap(default_labels_for_dataset("flower102", 3), paths["labelmap_path"])
    env_stamp = write_env_stamp(args, paths["env_stamp_path"], args.device)
    writer = SummaryWriter(log_dir=paths["tb_dir"])

    with torch.no_grad():
        train.valid(args, model, writer, loader, global_step=0, fiftyone_path=paths["fiftyone_path"])

    writer.close()

    assert paths["fiftyone_path"].exists()
    assert paths["labelmap_path"].exists()
    lines = paths["fiftyone_path"].read_text().strip().splitlines()
    assert len(lines) == 4
    assert "checksum" in label_info
    assert env_stamp["fp16"] is False
