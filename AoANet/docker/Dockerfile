FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

# Install some basic utilities
RUN apt-get -y update && \
    apt-get -y install \
        python3 \
        python3-dev \
        python3-pip \
        git \
        wget \
        sudo \
        curl \
        vim \
        unzip \
        cmake \
        build-essential \
        pkg-config \
        libatlas-base-dev \
        gfortran \
        libjasper-dev \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libv4l-dev \
        bzip2 \
        libx11-6 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*       


# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# All users can use /home/user as their home directory
ENV HOME=/home/
RUN chmod 777 /home/

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/miniconda/bin/conda clean -ya

# CUDA 10.0-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    "pytorch=1.2.0=py3.6_cuda10.0.130_cudnn7.6.2_0" \
    "torchvision=0.4.0=py36_cu100" \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests=2.19.1 \
 && conda clean -ya

# Install Graphviz
RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
 && conda clean -ya

RUN conda install -y scipy \
 && conda clean -ya

RUN conda install -y tensorboard \
 && conda clean -ya

RUN conda install -y python-lmdb \
 && conda clean -ya

RUN conda install -y scikit-image  \
 && conda clean -ya

RUN pip install gensim

RUN pip install pyemd

RUN pip install smart-open==1.9.0


RUN pip install transformers==2.11.0

RUN pip install tqdm
RUN pip install tensorboard
RUN pip install tensorboardX
RUN pip install pillow==6.1.0
RUN pip install bert_score
RUN pip install pandas


WORKDIR /work/recod/gabriel.santos/
RUN chmod -R 777  /etc/ /opt/ /root/ /home/ /mnt/ /usr/
