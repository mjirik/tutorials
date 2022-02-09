import detectron2
from detectron2.utils.logger import setup_logger

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from matplotlib import pyplot as plt
import skimage.io
from skimage import img_as_ubyte
import torch


# im = cv2.imread("image.jpg")

im = skimage.io.imread("http://images.cocodataset.org/val2017/000000439715.jpg")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

if not torch.cuda.is_available():
    cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)
# skimage use RGB colorspace, OpenCV and Detectron use BGR colorspace
im_bgr = img_as_ubyte(im)
outputs = predictor(im_bgr)

v = Visualizer(im_bgr[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
imout = out.get_image()[:, :, ::-1]
plt.imshow(imout)
plt.show()

plt.imsave("image.png", imout)