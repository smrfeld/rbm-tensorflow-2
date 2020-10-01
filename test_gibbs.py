import numpy as np
from rbm import GibbsSampler, RBM

no_visible = 10
no_hidden = 5

# RBM
rbm = RBM(
    no_visible=no_visible, 
    no_hidden=no_hidden
    )

# Gibbs sampler
g = GibbsSampler()

# Starting batch = random
batch_size = 3
visible_batch = np.random.randint(low=0, high=2, size=(batch_size, no_visible))
print("Starting visible:")
print(visible_batch)

hidden_batch = g.sample_hidden_given_visible(
    visible_batch=visible_batch,
    bias_hidden_tf=rbm.bias_hidden,
    weights_tf=rbm.weights,
    binary=True
    )
print("Visible -> hidden:")
print(hidden_batch)

visible_batch = g.sample_visible_given_hidden(
    hidden_batch=hidden_batch,
    bias_visible_tf=rbm.bias_visible,
    weights_tf=rbm.weights,
    binary=True
    )
print("Hidden -> visible:")
print(visible_batch)