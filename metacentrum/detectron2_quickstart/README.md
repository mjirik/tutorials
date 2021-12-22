# Learn about Metacentrum


[Metacentrum tutorial](https://wiki.metacentrum.cz/wiki/Pruvodce_pro_zacatecniky)



# Install detectron on Metacentrum

[Installation on Metacentrum is based on script by Jiri Vyskocil](https://github.com/vyskocj/ImageCLEFdrawnUI2021/blob/master/scripts/01_metacentrum_installation.sh)


This script contains only the installation procedure. User intervention may be required.

First you will need a computer with GPU. You will wait for it for few minutes.

```bash

# Interactive queue for 1 hour with 2 CPUs, 8 GB memory and 1 GPU
# (this installation works for "adan" ifiniband)
# NOTE: for running next step scripts (training the model), you can use the no interactive queue (without -I),
#       e.g.: qsub -l select=1:ncpus=2:mem=8gb:ngpus=1:cl_adan=True -l walltime=23:59:59 -q gpu 02_metacentrum_train_screenshot_sample.sh
qsub -I -l select=1:ncpus=2:mem=8gb:ngpus=1:cl_adan=True -l walltime=01:00:00 -q gpu
```

Run these lines on recived computer with GPU in interactive mode.
```bash
module add cuda-10.1
module add conda-modules-py37

module add gcc-8.3.0
module add ninja/ninja-1.10.0-gcc-8.3.0-xraep32

# create and activate "drawnUI-conda" environment in your home directory
conda create --prefix /storage/plzen1/home/$LOGNAME/.conda/envs/drawnUI-conda python=3.6
conda activate /storage/plzen1/home/$LOGNAME/.conda/envs/drawnUI-conda

# temp dir to not recieve error message: "Disk quota exceeded"
export TMPDIR=/storage/plzen1/home/$USER/condatmp
mkdir -p $TMPDIR

# installation of needed packages
conda install numpy                 # it is missing in conda at the default
conda install ninja                 # better to have it
pip install opencv-python           # if you need OpenCV in your scripts

# installation of needed packages for detectron2, We tested detectron2 version 0.5 with pytorch 1.7 and Cuda 10.1
mkdir -p extern
cd extern
git clone https://github.com/facebookresearch/detectron2.git                                                    # clone detectron2
cd detectron2
git checkout v0.5
cd ..
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html   # pytorch with cuda-10.1
pip install -e detectron2                                                                                       # detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html            # prebuild

# go to the local directory and clone the project
cd /storage/plzen1/home/$USER/
mkdir -p data

exit
```


Get the dataset 

```bash
mkdir -p ~data/cocos2d/orig/
cd ~/data/cocos2d/orig/
wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
unzip data.zip > /dev/null
```
