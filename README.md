# Deep Embedded Clustering

The original code for the DEC was from [piiswrong/dec](https://github.com/piiswrong/dec).

This package implements the algorithm described in paper "Unsupervised Deep Embedding for Clustering Analysis". It depends on opencv, numpy, scipy and Caffe.

## Usage
After cloning the repository, run `python main.py` with the command-line arguments.

### Docker

A Dockerfile has been provided to create a sterile development environment easily.  To build the environment, run `docker build --rm -t image_dec .` and then `docker run --rm -it image_dec /bin/bash` to shell into the running container.  Alternatively, [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) can be used to enable GPU capability.
