import numpy as np
from rbm import RBM

no_visible = 10
no_hidden = 5

# RBM
rbm = RBM(
    no_visible=no_visible, 
    no_hidden=no_hidden
    )

no_samples = 50
all_visible_samples = np.zeros((no_samples, no_visible))
for i in range(0,no_samples):
    all_visible_samples[i,0] = 1
    for j in range(1,no_visible):
        all_visible_samples[i,j] = abs(1-all_visible_samples[i,j-1])

print(all_visible_samples)

rbm.train(
    all_visible_samples=all_visible_samples,
    batch_size=5,
    no_cd_steps=2,
    no_iter=1000,
    learning_rate_weights=0.1,
    learning_rate_biases=0.01
)

# Sleep phase sample
visible_batch = np.random.randint(low=0,high=2,size=(1,no_visible))
visible_sampled, hidden_sampled = rbm.sample_model_dist(
    visible_batch=visible_batch,
    no_cd_steps=20
    )
print(visible_sampled)