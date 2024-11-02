## NIH
This document explains how to get the data, split the data into sets,
and preprocess the data in the common format:

### Setting up the environment TODO DOCKERIZE EVERYTHING

Before even attempting anything, we will use the environment provided by the paper,
which I updated (removed any conflicted packages with conda) before running inside the 
main directory of the repository:

Make sure conda is installed first TODO ADD INSTRUCTIONS

```bash
conda env create -f env.yml
```

Now activate the environment:
```bash
conda activate cxr_underdiag
```

You should see (cxr_underdiag) appear before your directory, if not try to
restart the terminal/make sure `conda` is properly installed before trying again, that may solve the problem.

### Getting the data from the website

I have already downloaded most of the important data in the 
repository/container, the only thing that needs help is the images. 
For that, please run the `batch_download_zips.py` script inside the `./images` directory.

For testing purposes I only downloaded the first batch of images with `single_download.py`

The images for NIH will be stored inside the `./images/images` directory.

### Organizing the data 



### Splitting the data

### Preprocessing the data
