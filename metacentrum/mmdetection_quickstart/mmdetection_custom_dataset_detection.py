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
logger = mmcv.utils.get_logger(name=__file__, log_level=logging.DEBUG)

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
model.show_result(img, result, out_file=outputdir / 'demo_output.jpg')# save image with result


# My dataset training
from mmcv import Config
cfg = Config.fromfile(mmdetection_path / 'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')

from mmdet.apis import set_random_seed

# Modify dataset type and path
cfg.dataset_type = 'CocoDataset'
cfg.data_root = input_data_dir / 'data/'
cfg.classes = ('date', 'fig', 'hazelnut',)

cfg.data.test.type = 'CocoDataset'
cfg.data.test.data_root = 'data/'
cfg.data.test.ann_file = 'trainval.json'
cfg.data.test.img_prefix = 'images/'
cfg.data.test.classes = cfg.classes

cfg.data.train.type = 'CocoDataset'
cfg.data.train.data_root = 'data/'
cfg.data.train.ann_file = 'trainval.json'
cfg.data.train.img_prefix = 'images/'
cfg.data.train.classes = cfg.classes

cfg.data.val.type = 'CocoDataset'
cfg.data.val.data_root = 'data/'
cfg.data.val.ann_file = 'trainval.json'
cfg.data.val.img_prefix = 'images/'
cfg.data.val.classes = cfg.classes

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 3
# If we need to finetune a model based on a pre-trained detector, we need to
# use load_from to set the path of checkpoints.
cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

# Set up working dir to save files and logs.
cfg.work_dir = outputdir / 'tutorial2_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
# cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')






logger.debug(f"outputdir={outputdir}")
logger.debug(f"input_data_dir={input_data_dir}")
# # print all files in input dir recursively to check everything
logger.debug(str(list(Path(input_data_dir)).glob("**/*")))


