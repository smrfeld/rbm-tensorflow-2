import tensorflow as tf
import numpy as np
from .gibbs_sampler import GibbsSampler

from typing import Tuple

def dot(a,b):
    return tf.reduce_sum( tf.multiply( a,b ), axis=1, keepdims=True ) 

def mat_vec(W, b):
    return tf.einsum('nm,m->n', W, b)

def vec_mat_vec(a, W, b):
    Wb = mat_vec(W,b)
    return tf.reduce_sum( tf.multiply( a, Wb ), axis=0, keepdims=True ) 

class RBM:

    def __init__(self, 
        no_visible : int, 
        no_hidden : int,
        std_dev_weights : float = 0.1,
        std_dev_biases : float = 0.01
        ):

        self.no_visible = no_visible
        self.no_hidden = no_hidden

        # The variables to minimize
        self.weights = tf.Variable(
            initial_value=tf.random.normal(
                shape=(no_visible,no_hidden),
                stddev=std_dev_weights,
                dtype='float32'
                ),
            name="weights"
            )
        self.bias_visible = tf.Variable(
            initial_value=tf.random.normal(
                shape=(1,no_visible),
                stddev=std_dev_biases,
                dtype='float32'
                ),
            name="bias visible"
            )
        self.bias_hidden = tf.Variable(
            initial_value=tf.random.normal(
                shape=(1,no_hidden),
                stddev=std_dev_biases,
                dtype='float32'
                ),
            name="bias hidden"
            )
        # print("Weights shape: " + str(self.weights.shape))
        # print("Bias shape: " + str(self.bias_visible.shape))
        # print("Bias shape: " + str(self.bias_hidden.shape))

        self.lr_weights = tf.Variable(initial_value=1.0,name="lr weights")
        self.lr_biases = tf.Variable(initial_value=1.0,name="lr biases")
        self.opt_weights = tf.keras.optimizers.Adam(learning_rate=self.lr_weights)
        self.opt_biases = tf.keras.optimizers.Adam(learning_rate=self.lr_biases)

        # Persistent chains
        self.visible_batch_asleep = []
        self.hidden_batch_asleep = []

    # Loss function
    def _loss_function(self, 
        visible_batch_awake : np.array, 
        hidden_batch_awake : np.array, 
        visible_batch_asleep : np.array, 
        hidden_batch_asleep : np.array):

        # Get batch size
        assert visible_batch_awake.shape[0] == hidden_batch_awake.shape[0]
        assert visible_batch_asleep.shape[0] == hidden_batch_asleep.shape[0]

        batch_size_awake = visible_batch_awake.shape[0]
        batch_size_asleep = visible_batch_asleep.shape[0]

        # Sum it up
        loss = 0.0

        # Asleep phase
        for i in range(0,batch_size_asleep):
            # Bias visible
            loss += dot( self.bias_visible, visible_batch_asleep[i] ) / batch_size_asleep
            # loss += tf.matmul(self.bias_visible, visible_batch_asleep[i]) / batch_size_asleep
            # Bias hidden
            loss += dot(self.bias_hidden, hidden_batch_asleep[i]) / batch_size_asleep
            # Weights 
            loss += vec_mat_vec(visible_batch_asleep[i], self.weights, hidden_batch_asleep[i].astype('float32')) / batch_size_asleep

        # Awake phase
        for i in range(0,batch_size_awake):
            # Bias visible
            loss -= dot(self.bias_visible, visible_batch_awake[i]) / batch_size_awake
            # Bias hidden
            loss -= dot(self.bias_hidden, hidden_batch_awake[i]) / batch_size_awake
            # Weights
            loss -= vec_mat_vec(visible_batch_awake[i], self.weights, hidden_batch_awake[i].astype('float32')) / batch_size_awake

        return loss

    def _awake_phase(self, 
        gibbs : GibbsSampler, 
        no_training_samples : int, 
        batch_size : int, 
        all_visible_samples : np.array) -> Tuple[np.array,np.array]:

        # Batch
        rand_sample = np.random.choice(no_training_samples,size=batch_size,replace=False)
        visible_batch_awake = all_visible_samples[rand_sample]
        
        # Binarize
        '''
        r = np.random.rand(visible_batch_awake.shape[0], visible_batch_awake.shape[1])
        visible_batch_awake[r < visible_batch_awake] = 1
        visible_batch_awake[r >= visible_batch_awake] = 0
        '''
        
        # Awake phase: activate hidden 
        hidden_batch_awake = gibbs.sample_hidden_given_visible(
            visible_batch=visible_batch_awake, 
            bias_hidden_tf=self.bias_hidden,
            weights_tf=self.weights, 
            binary=False
            )
        
        return (visible_batch_awake, hidden_batch_awake)

    def _asleep_phase(self,
        gibbs : GibbsSampler,
        persistent_cd : bool,
        visible_batch_awake : np.array,
        no_cd_steps : int):

        if not persistent_cd or len(self.visible_batch_asleep) == 0:
            self.visible_batch_asleep = visible_batch_awake.copy()
        for _ in range(0,no_cd_steps):
            self.hidden_batch_asleep = gibbs.sample_hidden_given_visible(
                visible_batch=self.visible_batch_asleep, 
                bias_hidden_tf=self.bias_hidden,
                weights_tf=self.weights, 
                binary=False
                )
            self.visible_batch_asleep = gibbs.sample_visible_given_hidden(
                hidden_batch=self.hidden_batch_asleep, 
                bias_visible_tf=self.bias_visible,
                weights_tf=self.weights, 
                binary=False
                )

    def train(self, 
        all_visible_samples : np.array, 
        batch_size : int, 
        no_cd_steps : int, 
        no_iter : int = 1000,
        learning_rate_weights : float = 0.1,
        learning_rate_biases : float = 0.01,
        persistent_cd : bool = True):
        
        # Set learning rate
        self.lr_weights.assign(learning_rate_weights)
        self.lr_biases.assign(learning_rate_biases)

        no_training_samples = all_visible_samples.shape[0]
        print("No training samples: %d" % no_training_samples)
        print("Batch size: %d" % batch_size)
        print("No CD steps: %d" % no_cd_steps)
    
        gibbs = GibbsSampler()

        # Loop
        self.opt_weights.iterations.assign(0)
        while self.opt_weights.iterations < no_iter:

            # Save the current value to compute the change later
            weights_last_step = self.weights.numpy()
            bias_visible_last_step = self.bias_visible.numpy()
            bias_hidden_last_step = self.bias_hidden.numpy()

            # Awake phase
            visible_batch_awake, hidden_batch_awake = self._awake_phase(
                gibbs=gibbs,
                no_training_samples=no_training_samples,
                batch_size=batch_size,
                all_visible_samples=all_visible_samples
                )

            # Asleep phase
            self._asleep_phase(
                gibbs=gibbs,
                persistent_cd=persistent_cd,
                visible_batch_awake=visible_batch_awake,
                no_cd_steps=no_cd_steps
                )
            
            # Train
            self.opt_weights.minimize(lambda: self._loss_function(
                visible_batch_awake=visible_batch_awake,
                hidden_batch_awake=hidden_batch_awake,
                visible_batch_asleep=self.visible_batch_asleep,
                hidden_batch_asleep=self.hidden_batch_asleep), 
                var_list=[
                    self.weights
                    ])
            self.opt_biases.minimize(lambda: self._loss_function(
                visible_batch_awake=visible_batch_awake,
                hidden_batch_awake=hidden_batch_awake,
                visible_batch_asleep=self.visible_batch_asleep,
                hidden_batch_asleep=self.hidden_batch_asleep), 
                var_list=[
                    self.bias_visible,
                    self.bias_hidden
                    ])
            
            # Calculate the change
            change_weights = abs(self.weights.numpy() - weights_last_step).mean()
            change_bias_visible = abs(self.bias_visible.numpy() - bias_visible_last_step).mean()
            change_bias_hidden = abs(self.bias_hidden.numpy() - bias_hidden_last_step).mean()

            # Report!
            print("Iteration: %d Change weights: %.16f bias visible: %.16f bias hidden: %.16f" % (
                self.opt_weights.iterations.numpy(), 
                change_weights,
                change_bias_visible,
                change_bias_hidden))

    def sample_model_dist(self, visible_batch : np.array, no_cd_steps : int):

        batch_size = visible_batch.shape[0]
        hidden_batch = np.zeros((batch_size,self.no_hidden))
        
        gibbs = GibbsSampler()

        # Binarize
        '''
        r = np.random.rand(visible_batch.shape[0], visible_batch.shape[1])
        visible_batch[r < visible_batch] = 1
        visible_batch[r >= visible_batch] = 0
        '''

        for _ in range(0,no_cd_steps):
            hidden_batch = gibbs.sample_hidden_given_visible(
                visible_batch=visible_batch, 
                bias_hidden_tf=self.bias_hidden,
                weights_tf=self.weights, 
                binary=False
                )
            
            visible_batch = gibbs.sample_visible_given_hidden(
                hidden_batch=hidden_batch, 
                bias_visible_tf=self.bias_visible,
                weights_tf=self.weights, 
                binary=False
                )

        return [visible_batch, hidden_batch]