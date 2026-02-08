# Use ubuntu 24.04 as an image
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies: gcc, ninja, tcsh, ssh, and other build essentials
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    python3.11-tk \
    python3-pip \
    python-is-python3 \
    gcc \
    g++ \
    vim \
    make \
    ninja-build \
    git \
    tcsh \
    pkg-config \
    python3-z3 \
    z3 \
    libz3-dev \
    libboost-python-dev \
    libgmp-dev \
    tk \
    tkcvs \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    x11-apps \
    x11-xserver-utils \
    tkcvs \
    x11vnc \
    xvfb \
    vim-gtk3 \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    autocutsel \
    tzdata \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install setuptools
RUN python3.11 -m pip install --no-cache-dir setuptools --target /usr/local/lib/python3.11/dist-packages

# Install Python packages including meson
RUN python3.11 -m pip install --ignore-installed --no-cache-dir -r requirements.txt

# Copy the wget  script
COPY python-wget.py .
# Make script executable
RUN chmod +x python-wget.py
#
# Copy the build script
COPY run_python_boost_build .
# Make script executable
RUN chmod +x run_python_boost_build
RUN ./run_python_boost_build
#
#Change python default version
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
# Copy the build script
COPY run_meson_build .
# Make script executable
RUN chmod +x run_meson_build
RUN ./run_meson_build
#
#Copy z3 python package
RUN cp -rp /usr/lib/python3/dist-packages/z3 /usr/local/lib/python3.11/dist-packages
#
#Workaround for TkAgg issue
COPY tkagg_patch.sh .
RUN chmod +x tkagg_patch.sh
RUN ./tkagg_patch.sh
#
#Mathsat
COPY run_mathsat_build .
RUN chmod +x run_mathsat_build
RUN ./run_mathsat_build
#tkdiff patch for https://bugs.launchpad.net/bugs/2139062
COPY run_tkdiff_patch .
RUN chmod +x run_tkdiff_patch
RUN ./run_tkdiff_patch
#UTF-8 fonts
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
#VNC
COPY start_vnc .
RUN chmod +x start_vnc
## Default command
CMD ["/bin/tcsh"]
