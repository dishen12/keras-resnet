#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
python ../cifar100_vgg_modify.py --only_last 1 --logPath "../logs/only_last/model1/"