FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG UID=1000
ARG GID=1000
ARG USERNAME=sahithi_kukkala
ARG CONDA_VERSION=py38_23.5.2-0

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PATH="/opt/conda/bin:$PATH" \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Validate UID and GID
RUN if [ -z "$UID" ] || [ -z "$GID" ]; then \
    echo "Error: UID and GID build arguments must be provided." >&2; \
    exit 1; \
    fi

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo git curl wget vim tzdata software-properties-common \
    build-essential gcc-10 g++-10 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 10 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$CONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Create user
RUN groupadd --gid ${GID} ${USERNAME} && \
    useradd --uid ${UID} --gid ${GID} -m -s /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Initialize Conda for bash
RUN conda init bash

CMD ["/bin/bash"]
