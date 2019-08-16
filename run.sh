#!/bin/sh

PYTHON=python
NETWORK_LIST="alexnet alexnet-v2 vgg16 vgg19 resnet101 resnet200 resnet_v2_50 resnet_v2_101 resnet_v2_200"
GPU=0
DTYPE=float32
BATCH_SIZE_LIST="2 4 8 16 32 64 128 256"
N_EPOCH=3
WARM_UP_NUM=3
DLLIB_LIST="pytorch tensorflow"

trap 'echo you hit Ctrl-C/Ctrl-\, now exiting..; pkill -P $$; exit' INT QUIT
for DLLIB in ${DLLIB_LIST}
do
    for NETWORK in ${NETWORK_LIST}
    do
        for BATCH_SIZE in ${BATCH_SIZE_LIST}
        do
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} inference_${DLLIB}.py --network ${NETWORK} \
                --dtype ${DTYPE} \
                --batch-size ${BATCH_SIZE} \
                --n-sample 2000 \
                --n-epoch ${N_EPOCH} \
                --warm-up-num ${WARM_UP_NUM} \
                --gpu 0
        done
    done
done

for NETWORK in ${NETWORK_LIST}
do
    ${PYTHON} plot_speed.py --network ${NETWORK}
done
