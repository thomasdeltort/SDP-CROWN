# SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming

This is an early release of the SDP-CROWN source code. We are in the process of  integrating SDP-CROWN into α,β-CROWN framework.

More details for SDP-CROWN can be found in our paper:

[SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming](https://arxiv.org/pdf/2506.06665)\
**ICML 2025**\
Hong-Ming Chiu, Hao Chen, Huan Zhang, Richard Y. Zhang

## Reproducing results

<p align="center">
<a href="https://abcrown.org"><img src="https://www.huan-zhang.com/images/upload/alpha-beta-crown/logo_2022.png" width="28%"></a>
</p>

### Installation and setup

Our code is tested on Python 3.11+ and PyTorch 2.3.1. It can be installed easily into a conda environment. If you don't have conda, you can install [miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
# Clone the alpha-beta-CROWN verifier
git clone https://github.com/huanzhang12/alpha-beta-CROWN.git
cd alpha-beta-CROWN
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
conda env create -f complete_verifier/environment.yml --name alpha-beta-crown  # install all dependents into the alpha-beta-crown environment
conda activate alpha-beta-crown  # activate the environment
```

This directory contains the following:\
`auto_LiRPA`: LiRPA with SDP-CROWN implementation.\
`models/` : Stores all pretrained model weights.\
`models.py` : Defines all models used in the SDP-CROWN paper.\
`utils.py` : Utility functions for loading models and datasets.\
`run_sdp_crown.sh` : Shell script with commands to reproduce the results in Table 1 of the SDP-CROWN paper.\
`sdp_crown.py` : Main script for running SDP-CROWN.
```
usage: sdp_crown.py [-h] [--radius RADIUS] [--lr_alpha LR_ALPHA] [--lr_lambda LR_LAMBDA] [--start START] [--end END]
                    [--model {mnist_mlp,mnist_convsmall,mnist_convlarge,cifar10_cnn_a,cifar10_cnn_b,cifar10_cnn_c,cifar10_convsmall,cifar10_convdeep,cifar10_convlarge}]

options:
  -h, --help             Show this help message and exit
  --radius RADIUS        L2 norm perturbation
  --lr_alpha LR_ALPHA    Alpha learning rate
  --lr_lambda LR_LAMBDA  Lambda learning rate
  --start START          Start index for the dataset
  --end END              End index for the dataset
  --model                Choose one of the predefined models
                         {mnist_mlp, mnist_convsmall, mnist_convlarge,
                          cifar10_cnn_a, cifar10_cnn_b, cifar10_cnn_c,
                          cifar10_convsmall, cifar10_convdeep, cifar10_convlarge}
```

### Reproducing SDP-CROWN results

To reproduce Talbe 1 of our paper, please run

```python
# MNIST Models
python sdp_crown.py --model mnist_mlp --radius 1.0
python sdp_crown.py --model mnist_convsmall --radius 0.3
python sdp_crown.py --model mnist_convlarge --radius 0.3
# CIRAR-10 Models
python sdp_crown.py --model cifar10_cnn_a --radius 24/255
python sdp_crown.py --model cifar10_cnn_b --radius 24/255
python sdp_crown.py --model cifar10_cnn_c --radius 24/255
python sdp_crown.py --model cifar10_convsmall --radius 24/255
python sdp_crown.py --model cifar10_convdeep --radius 24/255
python sdp_crown.py --model cifar10_convlarge --radius 8/255
```