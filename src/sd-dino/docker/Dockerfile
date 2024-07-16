FROM nvcr.io/nvidia/cuda:11.7.0-base-ubuntu20.04 AS sd-dino

USER root
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update 
RUN apt-get -y install pybind11-dev \
    git \
    wget \
    python3-tk \
    python3-pip 

RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html

COPY . /sd-dino

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN pip install pycocotools
RUN pip install cython

RUN pip install -e detectron2

WORKDIR /sd-dino
RUN pip install -e .

RUN ln -s /usr/bin/python3 /usr/bin/python

CMD ["/bin/bash"]


