#!/bin/bash

#################### FedAvg ####################

# nohup python -u train.py --method FedAvg --dataset CIFAR10 --non-iid --split-rule Dirichlet \
#     --split-coef 0.1 --active-ratio 0.1 --total-client 100 --comm-rounds 1000 \
#     --local-epochs 5 --batchsize 50 --local-learning-rate 0.1 --seed 0 \
#     --model ResNet18 > ./outputs/Cifar10-dir0.1_c100-0.1/FedAvg_ResNet18_seed0.out 2>&1 &



#################### FedDyn ####################

# nohup python -u train.py --method FedDyn --dataset CIFAR10 --non-iid --split-rule Dirichlet \
#     --split-coef 0.1 --active-ratio 0.1 --total-client 100 --comm-rounds 1000 \
#     --local-epochs 5 --batchsize 50 --local-learning-rate 0.1 --lr-decay 1 --seed 0 \
#     --model ResNet18 --lamb 0.1 > ./outputs/Cifar10-dir0.1_c100-0.1/FedDyn_lamb0.1_lrd1_ResNet18_seed0.out 2>&1 &



#################### FedFGAC ####################

nohup python -u train.py --dataset CIFAR10 --method FedFGAC --non-iid --split-rule Dirichlet \
    --split-coef 0.1 --active-ratio 0.1 --total-client 100 --comm-rounds 1000 \
    --local-epochs 5 --batchsize 50 --local-learning-rate 0.1 --seed 0 \
    --model ResNet18 > ./outputs/Cifar10-dir0.1_c100-0.1/FedFGAC_ResNet18_seed0.out 2>&1 &
