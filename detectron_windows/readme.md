
In February 2022 we experienced problems with installation. Check the [Known problems section](#known-problems)

# Install detectron2 with CPU only

[Install conda or miniconda](https://docs.conda.io/en/latest/miniconda.html)
```shell
conda create -n detectron python=3.8
conda activate detectron
conda install pytorch=1.10 torchvision torchaudio cpuonly -c pytorch
conda install cython opencv scikit-image
pip install git+https://github.com/facebookresearch/detectron2.git@v0.5
```

# Install detectron2 with GPU


[Install conda or miniconda](https://docs.conda.io/en/latest/miniconda.html)

```shell
conda create -n detectron python=3.8
conda activate detectron
conda install pytorch=1.10 torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install cython opencv scikit-image
pip install git+https://github.com/facebookresearch/detectron2.git@v0.5
```

Check the [pytorch web](https://pytorch.org/get-started/locally/)
for different Cuda or pytorch versions
 

# Test detectron

```shell
cd detectron_windows
conda activate detectron
python detectron2_demo.py
```

# Known problems

## WinError 182: Error loading "...\caffe2_detectron_ops_gpu.dll"

February 2022

Package `intel-openmp` might be already installed but it does not work properly

Solution
```shell
conda activate detectron
pip install intel-openmp
```


## ImportError: DLL load failed while importing win32file:

February 2022

Solution
```shell
conda activate detectron
conda install pywin32
```

[More details here](https://github.com/conansherry/detectron2/issues/8)