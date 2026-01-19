# TransFG: A Transformer Architecture for Fine-grained Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-stanford-1)](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford-1?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/image-classification-on-inaturalist)](https://paperswithcode.com/sota/image-classification-on-inaturalist?p=transfg-a-transformer-architecture-for-fine)

Official PyTorch code for the paper:  [*TransFG: A Transformer Architecture for Fine-grained Recognition (AAAI2022)*](https://arxiv.org/abs/2103.07976)  


## Framework

![](./TransFG.png)

## Environment (MacBook Air M2 / fp32)

- Python 3.10+
- PyTorch 2.x with MPS support (Metal) or CPU fallback
- torchvision 0.16+
- TensorBoard + FiftyOne for logging/inspection

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate transfg-mps
```

Or install via pip (arm64 macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Capture the environment hash for metadata (recommended):

```bash
conda env export --from-history > env_export.yml
shasum -a 256 env_export.yml > env_export.sha
```

## Usage
### 1. Download Google pre-trained ViT models

* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): ViT-B_16, ViT-B_32...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

In the paper, we use data from 5 publicly available datasets:

+ [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
+ [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
+ [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
+ [NABirds](http://dl.allaboutbirds.org/nabirds)
+ [iNaturalist 2017](https://github.com/visipedia/inat_comp/tree/master/2017)

Please download them from the official websites and put them in the corresponding folders.

### 3. Install required packages

Install dependencies with the following command:

```bash
pip3 install -r requirements.txt
```

### 4. Smoke train (flower102 tiny, fp32)

Single-device run on MacBook Air M2 (prefers MPS, falls back to CPU). Uses the flower102 tiny subsets (≤2 batches) for a fast smoke pass:

```bash
DATA_ROOT=${DATA_ROOT:-/path/to/data}
python train.py --name flower_mps \
  --dataset flower102 --data_root "$DATA_ROOT"/flower102 --model_type testing --img_size 64 \
  --num_steps 10 --eval_every 5 --train_batch_size 4 --eval_batch_size 2 \
  --output_dir output --prefer_mps \
  --tiny_train_subset flower102_tiny --tiny_infer_subset flower102_tiny
```

Artifacts: TensorBoard at `output/tb/flower_mps`, predictions at `output/flower_mps/fiftyone/predictions.jsonl`, labelmap at `output/flower_mps/labelmap.json`, env stamp at `output/flower_mps/env_stamp.json`, and retention note at `output/flower_mps/RETENTION.txt`.

### 5. Eval-only inference with prediction export

Run eval-only against an existing checkpoint:

```bash
DATA_ROOT=${DATA_ROOT:-/path/to/data}
python eval.py --name flower_eval \
  --dataset flower102 --data_root "$DATA_ROOT"/flower102 --img_size 64 --eval_batch_size 2 \
  --checkpoint output/flower_mps/checkpoints/ckpt.bin \
  --output_dir output --prefer_mps --tiny_infer_subset flower102_tiny
```

Artifacts: TensorBoard at `output/tb/flower_eval`, predictions at `output/flower_eval/fiftyone/predictions.jsonl`, labelmap at `output/flower_eval/labelmap.json`, env stamp at `output/flower_eval/env_stamp.json`, and retention note at `output/flower_eval/RETENTION.txt`.

### 6. Pytest smoke tests (tiny subsets)

Run the fast smoke suite (≤2 batches/split):

```bash
pytest -m "smoke or smoke_train or smoke_eval" -q
```

## Citation

If you find our work helpful in your research, please cite it as:

```
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}
```

## Acknowledgement

Many thanks to [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) for the PyTorch reimplementation of [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

