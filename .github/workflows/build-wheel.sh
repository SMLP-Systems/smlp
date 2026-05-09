#!/bin/bash

set -e -x

loc=$(dirname "${BASH_SOURCE[0]}")
project=$1
os=$2
tgt=$3

case "$os" in
ubuntu-*)
	dnf -y install \
		wget \
		gcc-toolset-13

	. /opt/rh/gcc-toolset-13/enable

	export R=$tgt

	source $loc/install-gmp.sh install

	source $loc/install-z3.sh install

	export GMP_ROOT=$gmp_R
	export Z3_PREFIX=$z3_R
	export LD_LIBRARY_PATH=$z3_R/lib:$gmp_R/lib

	;;
*)
	echo "error: OS unknown: $os" >&2
	exit 1
	;;
esac
