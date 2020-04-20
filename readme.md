# As stated previously, this implementation does not generate similar scores as the original tensorflow implementation. I recommend everyone to use mseitzer's implementation: https://github.com/mseitzer/pytorch-fid

# Fréchet Inception Distance (FID) for Pytorch

This implementation contains an implementation of the FID score in pytorch using a pre-trained InceptionV3 network. 

**NOTE** This is not the official implementation of FID. View [Two time-scale update rule for training GANs for the official implementation of FID](https://github.com/bioinf-jku/TTUR). This implementation uses [Pytorch pre-trained InceptionV3 network](https://pytorch.org/docs/stable/_modules/torchvision/models/inception.html#inception_v3) which is the same as the tensorflow pre-trained network.

## Fréchet Inception Distance
FID is a performance metric to evaluate the similarity between two dataset of images. It was introduced by the paper ["Two time-scale update rule for training GANs"](https://arxiv.org/abs/1706.08500). It is shown to correlate well to human evaluation of image quality, and it is able to detect intra-class mode collapse.


## Difference from the original tensorflow implementation
This implementation uses a different InceptionV3 model from what used in [the original implementation.](https://github.com/bioinf-jku/TTUR)
Not confirmed, but we believe pytorch uses the InceptionV3 network from 2016-08-28, while the original FID implementation uses a network from 2015-12-05. 


Original implementation InceptionV3 model:
- 2015-12-05, [code reference](https://github.com/bioinf-jku/TTUR/blob/8eca1abd808aba8d8cf90208887453744bb53190/fid.py#L263)

## Requirements

- python3 (not tested for python2)
- torch & torchvision 1.0 (might work for 0.4.0+)
- numpy
- scipy
- opencv


## Usage

To compute FID between to datasets, use the following command:
```bash
python fid.py --path1 path/to/real/data --path2 path/to/fake/data --batch-size 8
```
The script will use [all .png and .jpg files in the directories](fid.py#L244). It assumes the following:

The images are 

- unsigned int 8 between 0-255, or float32 between 0-1.
- all images are of same image shape

To compute InceptionV3 activations we:

- reshape images to (299,299). This is done in the official FID implementation
- normalize image similarly to [-1, 1]
- use network.eval() to use the running average for batch normalization.

## TODO:

- Port the InceptionV3 network used from the original implementation instead of using pytorch default.
