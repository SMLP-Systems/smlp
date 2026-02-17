# Use ubuntu 24.04 as an image
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Layer 1: base tools and Python PPA
RUN apt-get update && apt-get install -y \
    software-properties-common wget git gzip gcc g++ make pkg-config vim xvfb \
    && add-apt-repository ppa:deadsnakes/ppa -y

# Layer 2: Python 3.11
RUN apt-get install -y \
    tcsh z3 libz3-dev libgmp-dev ninja-build libboost-python-dev python3.11-dev python3.11-tk python3.11-venv python3.11

#Install Boost
ARG ARCH
RUN ARCH=$(uname -m) && \
    echo "Architecture: $ARCH" && \
    cd /tmp && \
    wget --tries=0 --read-timeout=20 --timeout=15 --waitretry=1 --retry-connrefused https://sourceforge.net/projects/boost/files/boost/1.83.0/boost_1_83_0.tar.gz && \
    tar -xvf boost_1_83_0.tar.gz  && \
    cd boost_1_83_0 && \
    ./bootstrap.sh --with-python=/usr/bin/python3.11 --with-libraries=python && \
    ./b2 install --prefix=$HOME/boost_py311 --with-python python=3.11 && \
    cp -p $HOME/boost_py311/lib/libboost_python311.a /usr/lib/$ARCH-linux-gnu && \
    cp -p $HOME/boost_py311/lib/libboost_python311.so.1.83.0 /usr/lib/$ARCH-linux-gnu && \
    ln -s /usr/lib/$ARCH-linux-gnu/{libboost_python311.so.1.83.0,libboost_python311.so} && \
    rm -rf /tmp/boost* /root/boost*
    
# Set working directory
WORKDIR /app

# Build SMLP and run DORA test
RUN touch /.dockerenv

#tkdiff patch for https://bugs.launchpad.net/bugs/2139062
COPY tkagg_patch.sh .
RUN chmod +x tkagg_patch.sh

RUN git clone https://github.com/smlp-systems/smlp smlp_venv && \
    cd smlp_venv && \
    git checkout remotes/origin/venv_build venv && \
    cd venv/python3.11 && \
    /app/tkagg_patch.sh && \
    ./run_venv_build

## Default command
CMD ["/bin/bash"]
