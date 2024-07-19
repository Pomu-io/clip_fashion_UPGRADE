ARG CUDA_VERSION=11.6.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ARG CAS_NAME=cas
WORKDIR /${CAS_NAME}

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# constant, won't invalidate cache
LABEL org.opencontainers.image.vendor="Jina AI Limited" \
      org.opencontainers.image.licenses="Apache 2.0" \
      org.opencontainers.image.title="CLIP-as-Service" \
      org.opencontainers.image.description="Embed images and sentences into fixed-length vectors with CLIP" \
      org.opencontainers.image.authors="hello@jina.ai" \
      org.opencontainers.image.url="clip-as-service" \
      org.opencontainers.image.documentation="https://clip-as-service.jina.ai/"

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip wget \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install --upgrade pip \
    && pip install wheel setuptools nvidia-pyindex

# Install PyTorch and other dependencies with specific versions for compatibility
RUN pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# Install additional dependencies for fashion-clip
RUN pip install boto3 appdirs pandas pyarrow matplotlib python-dotenv annoy transformers ipyplot datasets validators

# Install fashion-clip
RUN pip install fashion-clip

COPY server ./server
# given by builder
ARG PIP_TAG
RUN pip install --default-timeout=1000 --compile ./server/ \
    && if [ -n "${PIP_TAG}" ]; then pip install --default-timeout=1000 --compile "./server[${PIP_TAG}]" ; fi

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=${CAS_NAME}
ARG GROUP_NAME=${CAS_NAME}

RUN groupadd -g ${GROUP_ID} ${USER_NAME} &&\
    useradd -l -u ${USER_ID} -g ${USER_NAME} ${GROUP_NAME} &&\
    mkdir /home/${USER_NAME} &&\
    chown ${USER_NAME}:${GROUP_NAME} /home/${USER_NAME} &&\
    chown -R ${USER_NAME}:${GROUP_NAME} /${CAS_NAME}/

USER ${USER_NAME}

ENTRYPOINT ["python", "-m", "clip_server"]
