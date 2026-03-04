FROM quay.io/pypa/manylinux_2_28_x86_64

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
RUN dnf install -y \
        wget git make m4 pkg-config && \
    dnf clean all

# ---------------------------------------------------------------------------
# 2. Set Python 3.13 from pre-installed pypa versions
# ---------------------------------------------------------------------------
ENV PATH="/opt/python/cp313-cp313/bin:${PATH}"
ENV PYTHON=/opt/python/cp313-cp313/bin/python3.13
ENV PIP=/opt/python/cp313-cp313/bin/pip

# ---------------------------------------------------------------------------
# 3. Python build tools
# ---------------------------------------------------------------------------
RUN /opt/python/cp313-cp313/bin/pip install --upgrade pip && \
    /opt/python/cp313-cp313/bin/pip install setuptools==80.10.1 && \
    /opt/python/cp313-cp313/bin/pip install meson ninja z3-solver==4.8.12 auditwheel patchelf && \
    ln -sf /opt/python/cp313-cp313/bin/python3.13 /usr/local/bin/python3.13 && \
    ln -sf /opt/python/cp313-cp313/bin/pip /usr/local/bin/pip3.13 && \
    ln -sf /opt/python/cp313-cp313/bin/meson /usr/local/bin/meson && \
    ln -sf /opt/python/cp313-cp313/bin/ninja /usr/local/bin/ninja

# ---------------------------------------------------------------------------
# 4. Add ~/.local/bin to PATH
# ---------------------------------------------------------------------------
ENV PATH="/root/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# 5. Clone smlp and optionaly switch to branch
# ---------------------------------------------------------------------------
# Point setup.py to the correct z3 location for this Python installation
ENV Z3_PREFIX=/opt/python/cp313-cp313/lib/python3.13/site-packages/z3

WORKDIR /app
ARG GIT_BRANCH=smlp_python313
COPY run_git_clone .
RUN ./run_git_clone $GIT_BRANCH

# ---------------------------------------------------------------------------
# 6. Build wheel
# ---------------------------------------------------------------------------
WORKDIR smlp
RUN \cp -f manylinux_2_28/setup.py . && \
    /opt/python/cp313-cp313/bin/python3.13 -m pip wheel . -w dist/ && \
    /opt/python/cp313-cp313/bin/python3.13 manylinux_2_28/repair_wheel.py dist/ --plat manylinux_2_28_x86_64

CMD ["/usr/bin/bash"]
# The manylinux wheel is in /app/smlp/dist/
