#!/usr/bin/env python
# coding: utf-8

# ## Download YOLO5:
# - clone git repository
# - checkout specific version
# - eqivalent to execution of:
#     1. `cd beach-cleaning-object-detection/Notebooks/`
#     2. `git clone https://github.com/ultralytics/yolov5 ../yolov5`

# In[ ]:


# clone YOLOv5 repository (latest version):

get_ipython().system(u'git clone https://github.com/ultralytics/yolov5 ../yolov5  ')

# Original roboflow code took fixed old version, lets take the most up-to-date version:
# !git reset --hard 064365d8683fd002e9ad789c1e91fa3d021b44f0 ##TODO consider using latest version


# ## Install required python packages:
# - assumes `cd yolov5` was executed before
# - installs pytorch
# - installs requirements for yolo (takes a while, no status output, alternatively execute `pip install -r requirements.txt` in yolov5 folder)
# - Optional: Check file `yolov5/requirements.txt` there are some packages commented-out for optimization for nvidia export:
# 
# Diff:
# ```
# -# coremltools>=6.0  # CoreML export
# -# onnx>=1.12.0  # ONNX export
# -# onnx-simplifier>=0.4.1  # ONNX simplifier
# -# nvidia-pyindex  # TensorRT export
# -# nvidia-tensorrt  # TensorRT export
# +coremltools>=6.0  # CoreML export
# +onnx>=1.12.0  # ONNX export
# +onnx-simplifier>=0.4.1  # ONNX simplifier
# +nvidia-pyindex  # TensorRT export
# +nvidia-tensorrt  # TensorRT export
# ```
# 

# In[ ]:


# install dependencies as necessary
get_ipython().system(u'pip install -r ../yolov5/requirements.txt  # install dependencies (ignore errors)')
import torch

from IPython.display import Image, clear_output  # to display images

import sys
sys.path.append("../yolov5/") # add yolo code folder
from utils.downloads import attempt_download  # to download models/datasets

print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


# ## Fix compatibility issue with specific yolo version and pillow package:
# - Seems to be a version conflivt for git version 064365d8683fd002e9ad789c1e91fa3d021b44f0 of yolov5

# In[ ]:


get_ipython().system(u'pip install --force-reinstall -v "Pillow==9.5"')

