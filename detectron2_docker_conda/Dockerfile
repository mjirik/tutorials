FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

USER root

RUN apt-get update --yes
# RUN apt-get install --yes gpg vim ffmpeg libsm6 libxext6 curl supervisor multitail
RUN apt-get install --yes gpg vim ffmpeg curl multitail mc
# Install our public GPG key to trusted store
RUN curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor > conda.gpg
RUN install -o root -g root -m 644 conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg

# Add our Debian repo
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list
RUN apt-get update --yes
RUN apt-get install --yes conda

# make conda visible
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc
RUN /opt/conda/condabin/conda init bash
# RUN source /opt/conda/etc/profi le.d/conda.sh
ENV PATH="/opt/conda/condabin:${PATH}"
ARG PATH="/opt/conda/condabin:${PATH}"

USER appuser
RUN conda create -n cell_nuclei_app -c SimpleITK -c bioconda -c conda-forge python=3.6 openslide-python SimpleITK pandas loguru xlrd scipy=1.5 czifile opencv scikit-image
RUN conda install -n cell_nuclei_app -c SimpleITK -c bioconda -c conda-forge -c mjirik python=3.6 "io3d>=2.10" "imma>=0.18" "exsu>=0.7"
RUN conda run -n cell_nuclei_app --no-capture-output pip install pywebio
RUN conda list

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN conda run -n cell_nuclei_app --no-capture-output pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN conda run -n cell_nuclei_app --no-capture-output pip install --user torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# RUN conda run -n cell_nuclei_app --no-capture-output pip install --user torch==1.10+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN conda run -n cell_nuclei_app --no-capture-output pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
#RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN git clone https://github.com/facebookresearch/detectron2 --branch v0.5 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN conda run -n cell_nuclei_app --no-capture-output pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/detectron2_repo

WORKDIR /home/appuser
RUN git clone https://github.com/janburian/Web_App_PyWebIO
WORKDIR /home/appuser/Web_App_PyWebIO
RUN mkdir -p src/models
WORKDIR /home/appuser/Web_App_PyWebIO/src/models
RUN curl https://home.zcu.cz/~mjirik/lisa/modely_cell_nuclei/mask_rcnn_R_50_C4_3x.pth
RUN curl https://home.zcu.cz/~mjirik/lisa/modely_cell_nuclei/mask_rcnn_R_50_DC5_3x.pth
RUN curl https://home.zcu.cz/~mjirik/lisa/modely_cell_nuclei/mask_rcnn_R_50_FPN_3x.pth
RUN curl https://home.zcu.cz/~mjirik/lisa/modely_cell_nuclei/mask_rcnn_R_101_C4_3x.pth
RUN curl https://home.zcu.cz/~mjirik/lisa/modely_cell_nuclei/mask_rcnn_R_101_FPN_3x.pth
#COPY ../../mojevahy.pth bunky/vahy/

EXPOSE 64677
WORKDIR /home/appuser/Web_App_PyWebIO/src/

# odkomentovat
CMD python app.py

# zakomentovat zbytek
RUN conda run -n cell_nuclei_app pip install jupyterlab
EXPOSE 64677
WORKDIR /home/appuser
RUN pip install scikit-image io3d loguru pydicom imma SimpleITK

CMD jupyter lab --allow-root --ip=0.0.0.0 --port=64677

# docker build -t cell_nuclei_app .
# docker run -d -p 8888:8888 --name cell_nuclei_app cell_nuclei_app

# connection of hard drive volumes into docker image
# docker run -d -v "C:/Users/Jirik/projects/modely:/home/appuser/Web_App_PyWebIO" -v "C:/obrazky:/home/appuser" -p 8888:8888 --name cell_nuclei_app cell_nuclei_app
