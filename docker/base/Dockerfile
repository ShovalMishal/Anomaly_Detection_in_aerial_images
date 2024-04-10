FROM python:3.8
RUN pip install setuptools wheel
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install scikit-image scikit-learn plotly

# Install MMCV MMDetection
RUN pip install -U openmim && mim install mmengine && mim install "mmcv>=2.0.0rc4, <2.1.0"
 # Install MMRotate
RUN git clone --branch dev-3.x https://github.com/ShovalMishal/mmdection.git /mmdetection
RUN pip install --no-cache-dir -e /mmdetection
RUN git clone --branch dev-1.x https://github.com/ShovalMishal/mmrotate.git /mmrotate
RUN pip install --no-cache-dir -e /mmrotate

RUN pip install ipdb
RUN pip install transformers=="4.29.2"
RUN pip install pytorch-lightning=="2.1.2"
RUN apt-get update && apt-get install nano
RUN pip install tensorboard=="1.15.0"
RUN pip install numpy=="1.22.0"
RUN pip install protobuf=="3.20.3"

