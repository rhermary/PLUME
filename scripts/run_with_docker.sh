#!/bin/bash

IMAGE_NAME="wacv2025/id369"
SCRIPT="scripts/train_rcscombined_resnet.sh"
DEVICES=0
NORMAL_CLASSES=$(seq 0 9)

DEBUG=0
NB=$([[ $DEBUG -eq 0 ]] && echo $NB || echo 1)

XP_CONTAINER_DATA_DIR="/data/"
LOCAL_DATA_DIR="/home/rhermary/data/"

for normal_class in $NORMAL_CLASSES; do
    CONTAINER_ID=$(\
        docker create\
            --rm -it\
            --gpus all\
            --shm-size=10gb\
            --volume /etc/passwd:/etc/passwd:ro\
            --volume /etc/group:/etc/group:ro\
            --user $(id -u)\
            --mount type=bind,source=$LOCAL_DATA_DIR,target=$XP_CONTAINER_DATA_DIR\
            --cap-add=SYS_PTRACE\
            -e DATA_DIR=$XP_CONTAINER_DATA_DIR\
            -e NORMAL_CLASS=$normal_class\
            -e DEVICES=$DEVICES\
            $IMAGE_NAME bash $SCRIPT\
    )

    docker cp ./src $CONTAINER_ID:/app/
    docker cp . $CONTAINER_ID:/app/

    docker start $([[ $DEBUG -eq 1 ]] && echo "-ia") $CONTAINER_ID

    sleep $((10 + $RANDOM * 20 / 32767))
done
