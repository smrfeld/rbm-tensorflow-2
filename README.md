# RBMs revisited in TensorFlow 2

This repo. contains an implementation of RBMs in `TensorFlow 2`, (specifically: `2.3`).

Dreaming digits:
- Top row = original training data.
- Bottom row = dreamed digits after CD sampling.
<img src="figures_compiled/dreaming.png" alt="drawing" width="400"/>

Histogram of weights and biases:
<img src="figures_compiled/histograms.png" alt="drawing" width="400"/>

## Usage

```
python rbm_mnist.py
```
You will be prompted for more arguments.

## Tests

Tests for various components are:
```
python test_gibbs.py
python test_loss.py
python test_rbm.py
```

## Source

The implementation of the RBM (and the Gibbs sampler) is in the `rbm` folder.
