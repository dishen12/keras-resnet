#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
python cifar100_vgg_modify.py --input_layers 2 2 2 2 --logPath "./logs/2222/"