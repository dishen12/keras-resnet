#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
python ../cifar100_vgg_modify.py --input_layers 1 1 0 0 --logPath "./logs/11/"