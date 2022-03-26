# Learn about Metacentrum


[Metacentrum tutorial](https://wiki.metacentrum.cz/wiki/Pruvodce_pro_zacatecniky)

You can use WinSCP (for data upload) or Putty to connect to [available Metacentrum forntends](https://wiki.metacentrum.cz/wiki/Frontend)

# Metacentrum and `mmdetection`

We prepared *singularity* image with `mmdetection`. In the scripts we just use this image.


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
cd ~/projects/tutorials/metacentrum/mmdetection_quickstart/
qsub qsub_mmdetection_custom_dataset_detection.sh
```

Check the web if the task is running:
https://metavo.metacentrum.cz/pbsmon2/person

Check the logs are stored in `~/projects/tutorials/metacentrum/detectron2_quickstart/`
with the suffix `*.o*` and `*.e*`

# Debugging

It is easier to debug in interactive mode. Ask for interactive job. You will wait a while.

```bash

# Interactive queue for 1 hour with 2 CPUs, 8 GB memory and 1 GPU
# (this installation works for "adan" ifiniband)
# NOTE: for running next step scripts (training the model), you can use the no interactive queue (without -I),
#       e.g.: qsub -l select=1:ncpus=2:mem=8gb:ngpus=1:cl_adan=True -l walltime=23:59:59 -q gpu 02_metacentrum_train_screenshot_sample.sh
qsub -I -l select=1:ncpus=2:mem=8gb:ngpus=1:scratch_local=10gb:cl_adan=True -l walltime=01:00:00 -q gpu
```
Run your `.sh` script when your job is ready and you can write in the shell.
```bash
bash /storage/plzen1/home/$LOGNAME/projects/tutorials/metacentrum/mmdetection_quickstart/qsub_mmdetection_custom_dataset_detection.sh
```

