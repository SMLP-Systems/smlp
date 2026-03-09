FROM almalinux:9

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
RUN dnf install -y epel-release && \
    dnf install -y \
        ca-certificates \
        diffutils \
        git \
        gnupg2 \
        jq \
        libgomp \
        glibc-locale-source \
        glibc-langpack-en \
        gzip \
        tar \
        tcsh \
        tzdata \
        vim \
        wget \
        xorg-x11-fonts-Type1 \
        xorg-x11-fonts-misc \
        xorg-x11-server-Xvfb \
        xorg-x11-utils \
        procps-ng

# ---------------------------------------------------------------------------
# 2. Python
# ---------------------------------------------------------------------------

RUN dnf install -y \
        gcc \
        gcc-c++ \
        make \
        wget \
        openssl-devel \
        bzip2-devel \
        libffi-devel \
        zlib-devel \
        readline-devel \
        sqlite-devel \
        tk-devel && \
    dnf clean all

RUN wget https://www.python.org/ftp/python/3.13.12/Python-3.13.12.tgz && \
    tar -xf Python-3.13.12.tgz && \
    cd Python-3.13.12 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && rm -rf Python-3.13.12 Python-3.13.12.tgz

# ---------------------------------------------------------------------------
# 3. mathsat
# ---------------------------------------------------------------------------
WORKDIR /app
COPY run_mathsat_build.sh .
RUN ./run_mathsat_build.sh && rm -rf /tmp/mathsat*

# ---------------------------------------------------------------------------
# 4. Install pip and setuptools for Python 3.13
# ---------------------------------------------------------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 && \
    python3.13 -m pip install --upgrade setuptools

# ---------------------------------------------------------------------------
# 5. Clone smlp (ARG busts the cache to always get latest)
# ---------------------------------------------------------------------------
ARG CACHE_BUST_SMLP
COPY run_git_clone /app
RUN cd /app && ./run_git_clone smlp_python313

# ---------------------------------------------------------------------------
# 6. Install smlp from wheel
# ---------------------------------------------------------------------------
RUN python3.13 -m pip install /app/smlp/dist/smlp-0.1.0-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

# ---------------------------------------------------------------------------
# 7. Install pycaret from archive
# ---------------------------------------------------------------------------
RUN pip install /app/smlp/docker/python3.13/pycaret_313.tar.gz

# ---------------------------------------------------------------------------
# 8. Configure UTF-8 locale
# ---------------------------------------------------------------------------
RUN localedef -i en_US -f UTF-8 en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# ---------------------------------------------------------------------------
# 9. Docker GUI patch
# ---------------------------------------------------------------------------
COPY tkagg_patch.sh .
RUN ./tkagg_patch.sh

# ---------------------------------------------------------------------------
# 10. SMLP regression patch
# ---------------------------------------------------------------------------
COPY smlp_regr_patch.sh .
RUN ./smlp_regr_patch.sh

# ---------------------------------------------------------------------------
# 11. Define python version
# ---------------------------------------------------------------------------
RUN ln -sf /usr/local/bin/python3.13 /usr/bin/python && \
    ln -sf /usr/local/bin/python3.13 /usr/bin/python3

# ---------------------------------------------------------------------------
# 12. Copy and run DORA
# ---------------------------------------------------------------------------
COPY run_dora /app
RUN ./run_dora

# Default command
CMD ["/usr/bin/bash"]
