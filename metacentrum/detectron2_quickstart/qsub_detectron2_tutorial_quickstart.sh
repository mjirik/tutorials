#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_local=10gb
#PBS -l walltime=01:00:00 -q gpu

# modify/delete the above given guidelines according to your job's needs
# Please note that only one select= argument is allowed at a time.

# You can check available parameters and machines here:
# https://metavo.metacentrum.cz/pbsmon2/qsub_pbspro

# cl_konos=False:cl_gram=False

# # PBS -l select=1:ncpus=1:mem=1gb:scratch_local=4gb

# add to qsub with:
# qsub qsub_detectron2_tutorial_quickstart.sh

# nastaveni domovskeho adresare, v promenne $LOGNAME je ulozeno vase prihlasovaci jmeno
PROJECTDIR="/storage/plzen1/home/$LOGNAME/projects/tutorials/metacentrum/detectron2_quickstart"
DATADIR="/storage/plzen1/home/$LOGNAME/data/cocos2d/orig"
OUTPUTDIR="/storage/plzen1/home/$LOGNAME/data/cocos2d/processed"


echo "job: $PBS_JOBID running on: `uname -n`"


# nastaveni automatickeho vymazani adresare SCRATCH pro pripad chyby pri behu ulohy
trap 'clean_scratch' TERM EXIT

# vstup do adresare SCRATCH, nebo v pripade neuspechu ukonceni s chybovou hodnotou rovnou 1
cd $SCRATCHDIR || exit 1

# priprava vstupnich dat (kopirovani dat na vypocetni uzel)
mkdir -p $SCRATCHDIR/data/orig
mkdir -p $SCRATCHDIR/data/processed
mkdir -p $OUTPUTDIR
cp -r $DATADIR/* $SCRATCHDIR/data/orig/

echo "DATADIR=$DATADIR"
echo "ls DATADIR :"
ls $DATADIR

# list all files in SCRATCHDIR/data/orig
echo "SCRATCHDIR=$SCRATCHDIR"
echo "find SCRATCHDIR/data/orig/ :"
find $SCRATCHDIR/data/orig/

# spusteni aplikace - samotny vypocet

# activate environment option 1: miniconda installed
module add cuda-10.1
module add conda-modules-py37
module add gcc-8.3.0

#source conda activate drawnUI-conda
conda activate /storage/plzen1/home/$LOGNAME/.conda/envs/drawnUI-conda



#export PATH=/storage/plzen1/home/$LOGNAME/miniconda3/bin:$PATH
#source activate mytorch


# this is because of python click
export LC_ALL=C.UTF-8
export LANG=C.UTF-8


# Put your code here
python  $PROJECTDIR/detectron2_custom_coco_data_segmentation.py > results.txt

# kopirovani vystupnich dat z vypocetnicho uzlu do domovskeho adresare,
# pokud by pri kopirovani doslo k chybe, nebude adresar SCRATCH vymazan pro moznost rucniho vyzvednuti dat
cp results.txt $OUTPUTDIR || export CLEAN_SCRATCH=false
cp -r $SCRATCHDIR/data/processed/* $OUTPUTDIR/processed || export CLEAN_SCRATCH=false
find $OUTPUTDIR/processed/
