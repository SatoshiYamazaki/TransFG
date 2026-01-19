import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.modeling import CONFIGS, VisionTransformer
from utils.data_utils import SyntheticDataset
from train import valid


def test_model_forward_backward_cpu():
    config = CONFIGS["testing"]
    model = VisionTransformer(config, img_size=32, num_classes=3)
    model.train()

    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([1, 2])

    loss, logits = model(x, y)
    loss.backward()

    assert logits.shape[0] == 2
    assert logits.shape[1] == 3


def test_valid_writes_fiftyone(tmp_path: Path):
    config = CONFIGS["testing"]
    model = VisionTransformer(config, img_size=32, num_classes=3)
    model.to(torch.device("cpu"))

    dataset = SyntheticDataset(length=4, num_classes=3, image_size=(32, 32))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    args = types.SimpleNamespace(
        device=torch.device("cpu"),
        eval_batch_size=2,
        local_rank=-1,
        nprocs=1,
    )

    writer = SummaryWriter(log_dir=tmp_path / "tb")
    out_path = tmp_path / "predictions.jsonl"

    with torch.no_grad():
        valid(args, model, writer, loader, global_step=0, fiftyone_path=out_path)

    writer.close()

    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 4
