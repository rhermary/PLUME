#!/bin/bash

python -m tasks.train_features\
    --base_dir $DATA_DIR/results\
    --dataset_dir $DATA_DIR/datasets\
    --devices $DEVICES\
    --num_workers 5\
    --epochs 0\
    --batch_size 16\
    --acc_grad 2\
    --experiment features_resnet50\
    --model_name resnet50\
    --load_weights\
    --dataset cifar10

# With `--epochs 0`, no training will start but the network will be saved to be
# used later on as a not-retrained feature extractor.