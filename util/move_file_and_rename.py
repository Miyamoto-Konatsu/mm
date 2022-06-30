# -- coding: utf-8 --

import os
from shutil import copyfile

source_path = r'D:\engineering\software-engineering\data\images'
target_path = r'D:\Person_reID_baseline_pytorch-master\data\images\query'
for image, name in enumerate(os.listdir(source_path)):
    cls, id = name.split('.')[0].split('_')
    sub_dir = os.path.join(target_path, cls)
    source_name = os.path.join(source_path, name)
    target_name = id + '.jpg'
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    copyfile(source_name, sub_dir + '/' + target_name)

