import numpy as np
from rbm import RBM

no_visible = 10
no_hidden = 5

# RBM
rbm = RBM(
    no_visible=no_visible, 
    no_hidden=no_hidden
    )

# Batches
batch_size_awake = 3
batch_size_asleep = 4
visible_batch_awake = np.random.randint(low=0, high=2, size=(batch_size_awake, no_visible))
visible_batch_asleep = np.random.randint(low=0, high=2, size=(batch_size_asleep, no_visible))
hidden_batch_awake = np.random.randint(low=0, high=2, size=(batch_size_awake, no_hidden))
hidden_batch_asleep = np.random.randint(low=0, high=2, size=(batch_size_asleep, no_hidden))

res = rbm._loss_function(
    visible_batch_awake=visible_batch_awake,
    visible_batch_asleep=visible_batch_asleep,
    hidden_batch_awake=hidden_batch_awake,
    hidden_batch_asleep=hidden_batch_asleep   
    )
print(res)
