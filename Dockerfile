ARG DOCKER_IMAGE=archlinux:latest
FROM $DOCKER_IMAGE

#ENV CUDA_HOME=/opt/cuda
#ENV CUDA_TOOLKIT_ROOT_DIR=/opt/cuda
#ENV CUDACXX=/opt/cuda/bin/nvcc

LABEL author="Bensuperpc <bensuperpc@gmail.com>"
LABEL mantainer="Bensuperpc <bensuperpc@gmail.com>"

RUN pacman-key --init && pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    ffmpeg \
    cuda-tools \
    python \
    && pacman -Scc --noconfirm

RUN pacman -S --noconfirm \
    python-pip \
    python-tensorflow-opt-cuda \
    python-numpy \
    opencv-cuda \
    opencv-samples \
    python-pillow \
    python-matplotlib \
    && pacman -Scc --noconfirm

#RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY open_nsfw.py open_nsfw.py

ENTRYPOINT [ "python", "open_nsfw.py" ]
# CMD ["ct-ng"]
