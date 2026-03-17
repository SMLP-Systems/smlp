#!/bin/bash
set -e

MATHSAT="mathsat-5.6.8-linux-x86_64-reentrant"
MATHSAT_BIN_DIR="$(dirname "$(realpath "$0")")/external/${MATHSAT}/bin"

mkdir -p "${MATHSAT_BIN_DIR}"

wget --tries=5 --timeout=30 --waitretry=2 \
    "https://mathsat.fbk.eu/release/${MATHSAT}.tar.gz" \
    -O "/tmp/${MATHSAT}.tar.gz"

cd /tmp
tar -xvf "${MATHSAT}.tar.gz" > "${MATHSAT}.tar.log" 2>&1

cp -p "${MATHSAT}/bin/mathsat" "${MATHSAT_BIN_DIR}"
