#!/bin/bash


python -m tasks.train_rcscombined\
    --base_dir $DATA_DIR/results\
    --dataset_dir $DATA_DIR/datasets\
    --devices $DEVICES\
    --num_workers 3\
    --epochs 100\
    --batch_size 32\
    --acc_grad 1\
    --experiment train_plume_resnet50\
    --backbone_name resnet50backbone\
    --feature_extractor_checkpoint $DATA_DIR/resnet50-raw.ckpt\
    --normal_class $NORMAL_CLASS\
    --dataset oneclasscifar10\
    --lr "-1"\
    --classifier decclassifier1d\
    --gamma 1\
    --phi 0\
    --lambda 5\
    --eta 0\
    --tau 0.5

# To first save the ResNet50 backbone, see `save_resnet50.sh`
# To train without the contrastive loss (only linear noise): `--gamma 0`