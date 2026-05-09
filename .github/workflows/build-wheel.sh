#!/bin/bash

set -e -x

loc=$(dirname "$0")
project=$1
ghos=$2

dnf -y install wget
. /opt/rh/gcc-toolset-13/enable
source $loc/install-gmp.sh install
source $loc/install-z3.sh install
export GMP_ROOT=$gmp_R
export Z3_PREFIX=$z3_R
export LD_LIBRARY_PATH=$z3_R/lib:$gmp_R/lib
