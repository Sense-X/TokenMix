work_path=$(dirname $0)
filename=$(basename $work_path)
OMP_NUM_THREADS=1 \
srun -p $1 -n 8 --ntasks-per-node=8 --cpus-per-task=12 --gres=gpu:8 \
python -u main.py \
  --model deit_small \
  --input-size 224 \
  --batch-size 128 \
  --epochs 300 \
  --dist-eval \
  --drop-path 0.1 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --num_workers 12 \
  --use-bce \
  --output_dir ${work_path}/ckpt \
  --resume ${work_path}/ckpt/checkpoint.pth \
  --label_dir /path/to/label_top5_train_nfnet \
  --root_dir_train /path/to/imagenet/train \
  --root_dir_val /path/to/imagenet/val \
