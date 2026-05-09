#!/bin/bash

set -e -x

project=$1
loc="$project"/.github/workflows
ghos=$2

dnf -y install \
	wget \
	gcc-toolset-13

. /opt/rh/gcc-toolset-13/enable

source $loc/install-gmp.sh $loc install

source $loc/install-z3.sh $loc install

export GMP_ROOT=$gmp_R
export Z3_PREFIX=$z3_R
export LD_LIBRARY_PATH=$z3_R/lib:$gmp_R/lib
