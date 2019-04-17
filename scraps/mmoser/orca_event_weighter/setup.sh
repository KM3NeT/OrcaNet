#!/usr/bin/env bash

here=$PWD
oscprob_dir=$PWD/OscProb
if [[ ! -e $oscprob_dir/libOscProb.so ]]; then
    git clone https://github.com/joaoabcoelho/OscProb.git
    cd OscProb
    make
    cd $here
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$oscprob_dir
export PYTHONPATH=$PYTHONPATH:$here
