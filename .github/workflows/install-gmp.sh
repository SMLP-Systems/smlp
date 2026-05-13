#!/bin/bash

set -e -x

P=gmp

eval ${P}_V=${V:-6.3.0}					# version
eval ${P}_F=${F:-$P-\$`echo \${P}_V`.tar.xz}		# source archive name
eval ${P}_W=${W:-$HOME/$P}				# workdir root
eval ${P}_S=${S:-\$`echo ${P}_W`/$P-\$`echo \${P}_V`}	# source dir
eval ${P}_R=${R:-\$`echo ${P}_S`/prefix}		# install prefix

#eval ${P}_CPPFLAGS=-I\$`echo ${P}_R`/include
#eval ${P}_LDFLAGS=-L\$`echo ${P}_R`/lib
eval ${P}_PKG_CONFIG_PATH=\$`echo ${P}_R`/lib/pkgconfig

get() {
	wget -O $F https://ftpmirror.gnu.org/gnu/gmp/$F
}

unpack() {
	tar xfJ $F
}

configure() {
	cd $S
	./configure \
		--enable-cxx \
		--prefix=$R \
		--host=$HOST \
		CC=$CC \
		CXX=$CXX \
	&& :
}

install() {
	make -C $S -j`nproc` install
}

run() {
	eval local V=\${${P}_V}
	eval local F=\${${P}_F}
	eval local W=\${${P}_W}
	eval local S=\${${P}_S}
	eval local R=\${${P}_R}
	local T=$1

	mkdir -p $W

	local stages=( get unpack configure $T )
	local i f
	for i in `seq 1 ${#stages[@]}`; do
		local stage=${stages[i-1]}
		[[ -f $W/.$stage ]] && continue
		for f in $(echo ${stages[*]} | cut -d' ' -f$i-); do
			rm -f $W/.$f
		done
		cd $W
		$stage
		touch $W/.$stage
	done
}

if [ $# -eq 1 ]; then run $1; fi
