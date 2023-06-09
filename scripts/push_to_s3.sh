#/bin/bash

aws s3 sync data/ s3://app-id-108383-dep-id-105006-uu-id-ahc5pxjsqas8/omni-q/data/labs/

# Sync the fourier pickle files run with overlap as the objective function
aws s3 sync data_fourier/ s3://app-id-108383-dep-id-105006-uu-id-ahc5pxjsqas8/data/labs_fourier

# Sync the heatmap pickle files run with overlap as the objective function 
aws s3 sync data_overlap/ s3://app-id-108383-dep-id-105006-uu-id-ahc5pxjsqas8/data/labs_overlap
