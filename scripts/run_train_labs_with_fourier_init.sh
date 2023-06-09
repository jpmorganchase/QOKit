#!/bin/bash

parallel \
    --linebuffer \
    --jobs 6 \
    """
    python train_labs_with_fourier_init.py {1}
    """ ::: $(seq 18 23) 
