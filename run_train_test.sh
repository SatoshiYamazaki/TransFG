python train.py --name flower_mps \
 --dataset flowers-102 --data_root "$DATA_ROOT" --model_type ViT-B_16 --img_size 448 \
 --num_steps 2000 --eval_every 100 --train_batch_size 16 --eval_batch_size 16 \
 --output_dir output --prefer_mps \
 --pretrained_dir "$PRETRAINED_DIR"
