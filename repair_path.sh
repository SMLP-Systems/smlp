#!/usr/bin/env bash

set -x

SMLP="smlp"

SMLP_DIR=$(pwd)

#----- Fix libpython3.11.dylib path in the .so inside the whl into @rpath/libpython3.11.dylib

echo " Fixing path to libpython3.11.dylib in .so"

TMP_DIR=$(mktemp -d /tmp/smlp.XXXXXX)

ls $TMP_DIR

cp $SMLP_DIR/dist/smlp* $TMP_DIR/

for FILE_WHL in $TMP_DIR/*; do    
    echo "$FILE_WHL"
    base_name=${FILE_WHL##*/}
    echo "base_name: $base_name"
    based_name_no_ext="${base_name%.*}"
    cd "$TMP_DIR"

    mkdir "$based_name_no_ext"    
    unzip "$FILE_WHL" -d "$based_name_no_ext"
    
    cd "$based_name_no_ext/$SMLP"

    for FILE_SO in ./*.so; do
        echo "FILE_SO: $FILE_SO"
        install_name_tool -change libpython3.11.dylib @rpath/libpython3.11.dylib "$FILE_SO"
        codesign --force -s - "$FILE_SO"
    done    
    cd ../
    # in $based_name_no_ext
    zip -r "$FILE_WHL" ./*
    cp "$FILE_WHL" "$SMLP_DIR/dist"
    cd ../
    # in $TMP_DIR/
done

rm -r "$TMP_DIR"

cd $SMLP_DIR
