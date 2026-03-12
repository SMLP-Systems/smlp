FROM opensuse/leap:15.6

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
RUN zypper refresh && zypper install -y \
        curl \
        git \
        jq \
        libgomp1 \
        glibc-locale \
        glibc-locale-base \
        glibc-i18ndata \
        gzip \
        tar \
        tcsh \
        timezone \
        vim \
        wget \
        xorg-x11-fonts \
        xorg-x11-fonts-core \
        xorg-x11-server-Xvfb \
        xset \
        procps

RUN zypper addrepo https://download.opensuse.org/repositories/devel:languages:python:Factory/15.6/devel:languages:python:Factory.repo && \
    zypper --gpg-auto-import-keys refresh && \
    zypper install -y python312 python312-tk && \
    zypper clean --all

# ---------------------------------------------------------------------------
# 2. mathsat
# ---------------------------------------------------------------------------
WORKDIR /app
COPY run_mathsat_build.sh .
RUN ./run_mathsat_build.sh && rm -rf /tmp/mathsat*

# ---------------------------------------------------------------------------
# 3. Install pip and setuptools for Python 3.12
# ---------------------------------------------------------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    python3.12 -m pip install --upgrade setuptools

# ---------------------------------------------------------------------------
# 4. Clone smlp (ARG busts the cache to always get latest)
# ---------------------------------------------------------------------------
ARG CACHE_BUST_SMLP
COPY run_git_clone /app
RUN cd /app && ./run_git_clone smlp_python312

# ---------------------------------------------------------------------------
# 5. Install smlp from wheel
# ---------------------------------------------------------------------------
RUN python3.12 -m pip install /app/smlp/dist/smlp-0.1.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

# 6. Install pycaret from archive
# ---------------------------------------------------------------------------
RUN pip install /app/smlp/docker/python3.13/pycaret_313.tar.gz

# ---------------------------------------------------------------------------
# 7. Configure UTF-8 locale
# ---------------------------------------------------------------------------
RUN localedef -i en_US -f UTF-8 en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# ---------------------------------------------------------------------------
# 8. Docker GUI patch
# ---------------------------------------------------------------------------
COPY tkagg_patch.sh .
RUN ./tkagg_patch.sh

# ---------------------------------------------------------------------------
# 9. SMLP regression patch
# ---------------------------------------------------------------------------
COPY smlp_regr_patch.sh .
RUN ./smlp_regr_patch.sh

# ---------------------------------------------------------------------------
# 10. Define python version
# ---------------------------------------------------------------------------
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# ---------------------------------------------------------------------------
# 11. Copy and run DORA
# ---------------------------------------------------------------------------
COPY run_dora /app
RUN ./run_dora

# Default command
CMD ["/usr/bin/bash"]
