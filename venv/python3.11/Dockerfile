# Use ubuntu 24.04 as an image
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Layer 1: base tools and Python PPA
RUN apt-get update && apt-get install -y \
    software-properties-common wget git gzip vim xvfb \
    && add-apt-repository ppa:deadsnakes/ppa -y

# Layer 2: Python 3.11
RUN apt-get install -y \
    libgomp1 tcsh \
    python3.11-dev python3.11-tk python3.11-venv python3.11-distutils python3.11

# Set working directory
WORKDIR /app

# Build SMLP and run DORA test
RUN touch /.dockerenv

#tkdiff patch for https://bugs.launchpad.net/bugs/2139062
COPY tkagg_patch.sh .
RUN chmod +x tkagg_patch.sh

#Install smlp and run DORA test
COPY run_git_clone .
RUN  ./run_git_clone venv_build && \
     cd smlp/venv/python3.11 && \
     /app/tkagg_patch.sh && \
    ./run_venv_build

## Change working directory to venv root
WORKDIR /app/smlp/venv/python3.11

## Default command
CMD ["/bin/bash"]
