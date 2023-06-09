#!/bin/bash

set -e

for N in {31..33};
do
    chunk_width=10

    mkdir -p precomputed_energies_workspace/
    for chunk_id in {0..1023};
    do
        python precompute_and_cache_energies_one_chunk.py $N $chunk_id $chunk_width
    done
    python precompute_and_cache_energies_aggregate.py $N 1024 $chunk_width
    rm -r precomputed_energies_workspace/
done