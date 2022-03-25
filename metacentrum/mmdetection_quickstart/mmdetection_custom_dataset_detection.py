# Some basic setup

# import some common libraries
import logging

import matplotlib.pyplot as plt
import numpy as np
# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from pathlib import Path
mmdetection_path = Path(mmdet.__file__).parent.parent

import mmcv.utils
logger = mmcv.utils.get_logger(log_level=logging.DEBUG)

# import cv2
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
from pathlib import Path
import os
scratchdir = os.getenv('SCRATCHDIR', ".")
logname = os.getenv('LOGNAME', ".")
# from loguru import logger

input_data_dir = Path(scratchdir) / 'data/orig/'
outputdir = Path(scratchdir) / 'data/processed/'

import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector


# Choose to use a config and initialize the detector
config = mmdetection_path / 'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'
# Setup a checkpoint file to load
checkpoint = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

# Set the device to be used for evaluation
device='cuda:0'

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()

# Use the detector to do inference
img = mmdetection_path / 'demo/demo.jpg'
result = inference_detector(model, img)
model.show_result(img, result, out_file='demo_output.jpg')# save image with result










logger.debug(f"outputdir={outputdir}")
logger.debug(f"input_data_dir={input_data_dir}")
# # print all files in input dir recursively to check everything
logger.debug(str(Path(input_data_dir).glob("**/*")))


