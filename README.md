# antiprotonflux-neural-networks

This repository contains the code used my bachelor's thesis "Neural networks for event generation in astroparticle physics". A Python environment with the common used data science libraries (numpy, scipy, matplotlib, pandas), scikit-learn and tensorflow is needed to run this code.

The `ffnn_*.py` files are the feed forward neural networks with the respective varied parameters. The `rnn_*.py` files are the recurrent neural networks. 

`antiproton_flux.py` is the computation needed to fit the neural networks. You have to run 
```
$ python setup.py build_ext --inplace
```
to build the Cython file and be able to use it. If you have any trouble with Cython, you can replace 
```python
from antiproton_flux import dphidlogK
```
with 
```python
from antiproton_flux_python import dphidlogK
```
in the neural network files. This will use a pure python computation and is therefore slower.

The semi-analytical calculation can be found in https://arxiv.org/abs/1012.4515. The data used is taken from the website http://www.marcocirelli.net/PPPC4DMID.html.