# Use ubuntu 24.04 as an image
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Layer 1: base tools
RUN apt-get update && apt-get install -y \
    wget \
    vim \
    git \
    jq \
    tcsh \
    tzdata \
    locales \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Python 3.13
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg \
        curl \
        ca-certificates && \
        curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
            | gpg --dearmor -o /etc/apt/trusted.gpg.d/deadsnakes.gpg && \
        echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu noble main" \
            > /etc/apt/sources.list.d/deadsnakes.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-dev \
        python3.13-tk

# Layer 3: build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    ninja-build \
    pkg-config \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# Layer 4: Z3 and Boost
RUN apt-get update && apt-get install -y \
    python3-z3 \
    z3 \
    libz3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# Layer 5: X11 and GUI tools
RUN apt-get update && apt-get install -y \
    tk \
    tkcvs \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    x11-apps \
    x11-xserver-utils \
    x11vnc \
    xvfb \
    vim-gtk3 \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    autocutsel \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Copy the build script
COPY run_python_boost_build .

# Make script executable
RUN chmod +x run_python_boost_build
RUN ./run_python_boost_build && rm -rf /tmp/boost* /root/boost*

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

# Install Python packages including meson
RUN python3.13 -m pip install --ignore-installed --no-cache-dir -r requirements.txt

#Copy pycaret
COPY pycaret_313.tar.gz .

RUN  pip install --no-cache-dir pycaret_313.tar.gz && pip install numpy==2.4.1 && rm -f pycaret_313.tar.gz

ARG CACHE_BUST_SMLP

#Change python default version
RUN ln -sf /usr/bin/python3.13 /usr/bin/python3
RUN ln -sf /usr/bin/python3.13 /usr/bin/python

# Copy the build script
COPY run_meson_build .
# Make script executable
RUN chmod +x run_meson_build
RUN ./run_meson_build

#Copy z3 python package
RUN cp -rp /usr/lib/python3/dist-packages/z3 /usr/local/lib/python3.13/dist-packages

#Workaround for TkAgg issue
COPY tkagg_patch.sh .
RUN chmod +x tkagg_patch.sh
RUN ./tkagg_patch.sh

#Mathsat
COPY run_mathsat_build .
RUN chmod +x run_mathsat_build
RUN ./run_mathsat_build && rm -rf /tmp/mathsat*

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
