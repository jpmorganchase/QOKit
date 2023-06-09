#/bin/bash

aws s3 sync s3://app-id-108383-dep-id-105006-uu-id-ahc5pxjsqas8/omni-q/data/labs/ data/

# Has the fourier pickle files run with overlap as the objective function
aws s3 sync s3://app-id-108383-dep-id-105006-uu-id-ahc5pxjsqas8/data/labs_fourier data_fourier/

# Has the heatmap pickle files run with overlap as the objective function 
aws s3 sync s3://app-id-108383-dep-id-105006-uu-id-ahc5pxjsqas8/data/labs_overlap data_overlap/ 
