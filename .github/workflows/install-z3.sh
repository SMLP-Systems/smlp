#!/bin/bash

set -e -x

source "$(dirname "${BASH_SOURCE[0]}")"/install-gmp.sh

P=z3

eval ${P}_V=${V:-4.8.12}					# version
eval ${P}_F=${F:-$P-\$`echo \${P}_V`.tar.xz}			# source archive name
eval ${P}_W=${W:-$HOME/$P}					# workdir root
eval ${P}_S=${S:-\$`echo ${P}_W`/$P-$P-\$`echo \${P}_V`}	# source dir
eval ${P}_R=${R:-\$`echo ${P}_S`/prefix}			# install prefix

export PYTHON=python3.11
export CPPFLAGS+=" -I$gmp_R/include"
export LDFLAGS+=" -L$gmp_R/lib -Wl,-rpath,$gmp_R/lib"
export PKG_CONFIG_PATH=$gmp_PKG_CONFIG_PATH${PKG_CONFIG_PATH:+:}$PKG_CONFIG_PATH

run() {
	eval local V=\${${P}_V}
	eval local F=\${${P}_F}
	eval local W=\${${P}_W}
	eval local S=\${${P}_S}
	eval local R=\${${P}_R}
	local T=$1
	local B=build

	mkdir -p $W
	cd $W

	# get
	if ! [ -f $W/.get ]; then
		rm -f $W/.{unpack,configure,$T}
		wget -O $F https://github.com/Z3Prover/z3/archive/refs/tags/$P-$V.tar.gz
		touch $W/.get
	fi

	# unpack
	if ! [ -f $W/.unpack ]; then
		rm -f $W/.{configure,$T}
		tar xfz $F
		touch $W/.unpack
	fi

	# configure
	if ! [ -f $W/.configure ]; then
		rm -f $W/.$T
		cd $S && ./configure \
			--gmp \
			--build=$B \
			--prefix=$R \
		&& cd $W
		touch $W/.configure
	fi

	# build/install
	if ! [ -f $W/.$T ]; then
		sed -i.bak 's/@//' $S/$B/Makefile	# want to see commands
		env LIBRARY_PATH=$gmp_R/lib make -C $S/$B -j`nproc` $T
		touch $W/.$T
	fi
}

if [ $# -eq 1 ]; then run $1; fi
