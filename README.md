# Face Mask detection

To train this model, we use the data from the publicly available repository https://github.com/prajnasb/observations.git

## Requirements

If you have anaconda installed, you can run from the root of this repository:

    conda env update -f environment-cpu.yml # if you don't have GPUs
    conda activate facemask

This will create a `facemask` environment with all the dependencies installed.

## Download data

To train the model you need to download the data using dvc

    dvc pull
