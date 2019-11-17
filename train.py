import os
import sys
import random
import math
import re
import time
import numpy as np
import pandas as pd
from models import box_cpn as cpn
import configs
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, dest='model_path', default='data/pretrain/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
parser.add_argument('--cfg', '-c', type=str, dest='cfg', default="configs/BOX_CPN_ResNet50_FPN_cfg.py")
parser.add_argument('--data', '-d', type=str, dest='data_path', default="./data/annotation6.csv")
args = parser.parse_args()


# Root directory of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
model_path = args.model_path
logs = MODEL_DIR
config_file = os.path.basename(args.cfg.split('.')[0])
config_def = eval('configs.' + config_file + '.Config')
config = config_def()
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUs

print("Model: ", model_path)
print("Logs: ", logs)
config.display()

model = cpn.CPN(mode="training", config=config, model_dir=logs)

# Select weights file to load
if model_path.lower() == "last":
    # Find last trained weights
    model_path = model.find_last()[1]

# Load weights
print("Loading weights ", model_path)
model.load_weights(model_path, by_name=True)
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
tf.Session(config=config_tf)

df = pd.read_csv(args.data_path, index_col=0)
dataset_train, dataset_val = train_test_split(df, test_size=0.3, random_state=0)

# Training
base_lr = config.LEARNING_RATE
for i in range(0, 10):
    model.train(dataset_train,
                dataset_val,
                learning_rate=base_lr,
                epochs=10 * (i + 1),
                layers='all')
    base_lr = base_lr / 2
