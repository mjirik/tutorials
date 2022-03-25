# Learn about Metacentrum


[Metacentrum tutorial](https://wiki.metacentrum.cz/wiki/Pruvodce_pro_zacatecniky)

You can use WinSCP (for data upload) or Putty to connect to [available Metacentrum forntends](https://wiki.metacentrum.cz/wiki/Frontend)

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
conda create --yes --prefix /storage/plzen1/home/$LOGNAME/.conda/envs/drawnUI-conda python=3.6
conda activate /storage/plzen1/home/$LOGNAME/.conda/envs/drawnUI-conda

# temp dir to not recieve error message: "Disk quota exceeded"
export TMPDIR=/storage/plzen1/home/$USER/condatmp
mkdir -p $TMPDIR

# installation of needed packages
#    numpy - it is missing in conda at the default
#    ninja - better to have it
#    loguru - easier logging
#    ...if you need more packages your scripts, here is the place
conda install --yes -c conda-forge numpy ninja loguru

pip install opencv-python           

# installation of needed packages for detectron2, We tested detectron2 version 0.5 with pytorch 1.7 and Cuda 10.1
git clone https://github.com/facebookresearch/detectron2.git  --branch v0.5                                              

pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html   # pytorch with cuda-10.1
pip install -e detectron2                                                                                       # detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html            # prebuild

exit
```


Get the dataset 

```bash
mkdir -p ~/data/cocos2d/orig/
cd ~/data/cocos2d/orig/
wget https://github.com/Tony607/detectron2_instance_segmentation_demo/releases/download/V0.1/data.zip
unzip data.zip > /dev/null
```


Get the scripts to run
```bash
mkdir -p ~/projects
cd ~/projects/
git clone https://github.com/mjirik/tutorials.git
```


Run the experiment
```bash
cd ~/projects/tutorials/metacentrum/detectron2_quickstart/
qsub qsub_detectron2_tutorial_quickstart.sh
```

Check the web if the task is running:
https://metavo.metacentrum.cz/pbsmon2/user/mjirik

Check the logs are stored in `~/projects/tutorials/metacentrum/detectron2_quickstart/`
with the suffix `*.o*` and `*.e*`

The output files
```bash

```


and check the logs
