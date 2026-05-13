#!/bin/bash

set -e -x

loc=$(dirname "${BASH_SOURCE[0]}")	# directory of this script
project=$1				# project source directory
label=$2				# Github runner label
tgt=$3					# prefix to install packages to

case "$label" in
ubuntu-*)
	# This is executed inside a manylinux Docker image. In this case,
	# manylinux_2_28 based on almalinux-8.10.
	dnf -y install \
		wget \
		gcc-toolset-13

	echo "source /opt/rh/gcc-toolset-13/enable" >> $HOME/.bashrc
	source /opt/rh/gcc-toolset-13/enable

	export R=$tgt
	export HOST=$(uname -m)-pc-linux-gnu

	source $loc/install-gmp.sh install

	source $loc/install-z3.sh install

	find $tgt

	;;
macos-26*)
	export CC=gcc-13
	export CXX=g++-13
	export R=$tgt
	export HOST=$(uname -m)-apple-darwin25.3.0

	source $loc/install-gmp.sh install

	source $loc/install-z3.sh install

	;;
*)
	echo "error: OS unknown: $os" >&2
	exit 1
	;;
esac
