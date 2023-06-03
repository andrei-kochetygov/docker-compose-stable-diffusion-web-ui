FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_MODULE_LOADING=LAZY
ARG NUMEXPR_MAX_THREADS=16
ARG PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y git-lfs python3.10 python3.10-dev python3-pip wget python3-opencv \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip \
    && pip3 install fastapi opencv-python \
    && git lfs install \
    && git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git ./

EXPOSE 7860

ENTRYPOINT [ "python3", "launch.py", "--opt-sdp-attention", "--upcast-sampling", "--opt-channelslast", "--xformers", "--opt-sub-quad-attention" ]
