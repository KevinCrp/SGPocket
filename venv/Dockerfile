FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
WORKDIR "/home"

# Add missing Public Keys
RUN apt-get update 2>&1 | sed -ne 's?^.*NO_PUBKEY ??p' | xargs -r -- apt-key adv --keyserver keyserver.ubuntu.com --recv-keys

# Install base utilities
RUN apt-get update && apt-get install -y build-essential \
    autoconf \
    htop \
    nano \
    wget \
    unzip \
    git \
    dssp
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

WORKDIR "/workspace"

RUN conda install scikit-learn-intelex=2023.0.2 -y
RUN conda install pandas=1.5.3 -y
RUN conda install biopandas=0.4.1 -c conda-forge -y
RUN conda install pyg=2.3.0 -c pyg -y
RUN conda install pytorch-lightning=2.0.2 -c conda-forge -y
# RUN conda install matplotlib=3.7.1 -c conda-forge -y
# RUN conda install plotly=5.13.1 -c plotly -y 
# RUN conda install seaborn=0.12.2 -c conda-forge -y
RUN conda install scipy=1.10.0 -c anaconda -y
RUN conda install -c conda-forge tensorboard -y

RUN pip install biopython==1.81
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
