# TransFG: A Transformer Architecture for Fine-grained Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-stanford-1)](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford-1?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/image-classification-on-inaturalist)](https://paperswithcode.com/sota/image-classification-on-inaturalist?p=transfg-a-transformer-architecture-for-fine)

Official PyTorch code for the paper:  [*TransFG: A Transformer Architecture for Fine-grained Recognition (AAAI2022)*](https://arxiv.org/abs/2103.07976)  


## Framework

![](./TransFG.png)

## Environment (MacBook Air M2 / modern stack)

- Python 3.10+
- PyTorch 2.x with MPS support (Metal) or CPU
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

### 4. Train (MPS/CPU friendly)

Single-device run on MacBook Air M2 (MPS) for smoke validation (uses synthetic tiny data):

```bash
python train.py --name smoke_mps --dataset synthetic --model_type testing --img_size 64 \
  --num_steps 10 --eval_every 5 --train_batch_size 4 --eval_batch_size 2
```

Full run on real data (update paths accordingly):

```bash
python train.py --name cub_run --dataset CUB_200_2011 --data_root /path/to/data \
  --pretrained_dir /path/to/ViT-B_16.npz --num_steps 10000 --eval_every 500 \
  --train_batch_size 16 --eval_batch_size 8 --amp
```

TensorBoard logs are written to `output/tb/<run_name>` and a final FiftyOne-compatible
prediction file to `output/<run_name>/fiftyone/predictions.jsonl`.

### 5. Eval-only / inference with prediction export

Given a fine-tuned checkpoint:

```bash
python train.py --name cub_eval --dataset CUB_200_2011 --data_root /path/to/data \
  --pretrained_dir /path/to/ViT-B_16.npz --checkpoint output/cub_run_checkpoint.bin \
  --eval_only --eval_batch_size 8
```

The eval run writes metrics to TensorBoard and predictions to
`output/cub_eval/fiftyone/predictions.jsonl`.

### 6. Pytest smoke tests (tiny subsets)

Run the fast smoke suite (≤2 batches/split, synthetic data):

```bash
pytest -m "not slow" tests/test_smoke.py
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

