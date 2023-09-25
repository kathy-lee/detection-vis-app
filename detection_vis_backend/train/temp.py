import subprocess
import logging
import json
import os
from metaflow import Flow

n_frame = 897
win_size = 16
step = 1
stride = 4

n_data_in_seq = (n_frame - (win_size * step - 1)) // stride + (
                    1 if (n_frame - (win_size * step - 1)) % stride > 0 else 0)

model_rootdir = "/home/kangle/dataset/trained_models"
model_chosen = "FFTRadNet___Aug-03-2023___19:57:20"
model_path = os.path.join(model_rootdir, model_chosen)

# in default choose the last epoch checkpoint model
files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
print(files)
checkpoint_count = len(files)-4
b = list(range(1,checkpoint_count+1))
print(b)