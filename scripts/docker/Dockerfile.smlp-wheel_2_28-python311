FROM quay.io/pypa/manylinux_2_28_x86_64

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
RUN dnf install -y \
        wget git make m4 pkg-config && \
    dnf clean all

# ---------------------------------------------------------------------------
# 2. Set Python 3.11 from pre-installed pypa versions
# ---------------------------------------------------------------------------
ENV PATH="/opt/python/cp311-cp311/bin:${PATH}"
ENV PYTHON=/opt/python/cp311-cp311/bin/python3.11
ENV PIP=/opt/python/cp311-cp311/bin/pip

# ---------------------------------------------------------------------------
# 3. Python build tools
# ---------------------------------------------------------------------------
RUN /opt/python/cp311-cp311/bin/pip install --upgrade pip && \
    /opt/python/cp311-cp311/bin/pip install setuptools==71.1.0 && \
    /opt/python/cp311-cp311/bin/pip install meson ninja z3-solver==4.8.12 auditwheel patchelf && \
    ln -sf /opt/python/cp311-cp311/bin/python3.11 /usr/local/bin/python3.11 && \
    ln -sf /opt/python/cp311-cp311/bin/pip /usr/local/bin/pip3.11 && \
    ln -sf /opt/python/cp311-cp311/bin/meson /usr/local/bin/meson && \
    ln -sf /opt/python/cp311-cp311/bin/ninja /usr/local/bin/ninja

# ---------------------------------------------------------------------------
# 4. Add ~/.local/bin to PATH
# ---------------------------------------------------------------------------
ENV PATH="/root/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# 5. Clone smlp and optionaly switch to branch
# ---------------------------------------------------------------------------
# Point setup.py to the correct z3 location for this Python installation
ENV Z3_PREFIX=/opt/python/cp311-cp311/lib/python3.11/site-packages/z3/lib

WORKDIR /app
COPY run_git_clone .
ARG CACHE_BUST_SMLP
ARG GIT_BRANCH=master
RUN echo "Building image for branch: $GIT_BRANCH"
RUN ./run_git_clone $GIT_BRANCH

# ---------------------------------------------------------------------------
# 6. Build wheel
# ---------------------------------------------------------------------------
WORKDIR smlp
RUN /opt/python/cp311-cp311/bin/python3.11 -m pip -v wheel . -w dist/ && \
    /opt/python/cp311-cp311/bin/python3.11 repair_wheel.py dist/ --plat manylinux_2_28_x86_64

CMD ["/usr/bin/bash"]
# The manylinux wheel is in /app/smlp/dist/
