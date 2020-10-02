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
        """Constructor

        Args:
            no_visible (int): No visible units
            no_hidden (int): No hidden units
            std_dev_weights (float, optional): Standard deviation for weights. Defaults to 0.1.
            std_dev_biases (float, optional): Standard deviation for biases. Defaults to 0.01.
        """

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

        self.lr_weights = tf.Variable(initial_value=1.0,name="lr weights")
        self.lr_biases = tf.Variable(initial_value=1.0,name="lr biases")
        self.opt_weights = tf.keras.optimizers.Adam(learning_rate=self.lr_weights)
        self.opt_biases = tf.keras.optimizers.Adam(learning_rate=self.lr_biases)

        # Persistent chains
        self.visible_batch_asleep = []
        self.hidden_batch_asleep = []

    def _loss_function(self, 
        visible_batch_awake : np.array, 
        hidden_batch_awake : np.array, 
        visible_batch_asleep : np.array, 
        hidden_batch_asleep : np.array):
        """Loss function ~ See docs! This is an approximation that only accurately reproduces the first order gradients.

        Args:
            visible_batch_awake (np.array): Visible batch in awake phase of size (batch_size_awake, no_visible)
            hidden_batch_awake (np.array): Hidden batch in awake phase of size (batch_size_awake, no_hidden)
            visible_batch_asleep (np.array): Visible batch in asleep phase of size (batch_size_asleep, no_visible)
            hidden_batch_asleep (np.array): Hidden batch in asleep phase of size (batch_size_asleep, no_hidden)

        Returns:
            float: Loss function value ~ See docs! This only accurately reproduces the first order gradients.
        """

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
        """Awake phase

        Args:
            gibbs (GibbsSampler): Gibbs Sampler
            no_training_samples (int): No training samples
            batch_size (int): Batch size
            all_visible_samples (np.array): All visible samples in shape (no_samples, no_visible)

        Returns:
            Tuple[np.array,np.array]: (Visible batch in shape (batch_size,no_visible), Hidden batch in shape (batch_size, no_hidden))
        """

        # Batch
        rand_sample = np.random.choice(no_training_samples,size=batch_size,replace=False)
        visible_batch_awake = all_visible_samples[rand_sample]

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
        hidden_batch_awake : np.array,
        no_cd_steps : int):
        """Asleep phase

        Args:
            gibbs (GibbsSampler): Gibbs sampler
            persistent_cd (bool): True for persistent CD
            hidden_batch_awake (np.array): Hidden batch from the awake phase in shape (batch_size_awake, no_hidden)
            no_cd_steps (int): No CD steps
        """

        if not persistent_cd or len(self.visible_batch_asleep) == 0:
            self.hidden_batch_asleep = np.copy(hidden_batch_awake)
        for i in range(0,no_cd_steps):

            if i == no_cd_steps - 1:
                binary = True
            else:
                binary = False
            self.visible_batch_asleep = gibbs.sample_visible_given_hidden(
                hidden_batch=self.hidden_batch_asleep, 
                bias_visible_tf=self.bias_visible,
                weights_tf=self.weights, 
                binary=binary
                )

            self.hidden_batch_asleep = gibbs.sample_hidden_given_visible(
                visible_batch=self.visible_batch_asleep, 
                bias_hidden_tf=self.bias_hidden,
                weights_tf=self.weights, 
                binary=False
                )

    def train(self,
        epoch_idx : int,
        all_visible_samples : np.array, 
        batch_size : int, 
        no_cd_steps : int, 
        no_iter : int = 1000,
        learning_rate_weights : float = 0.1,
        learning_rate_biases : float = 0.01,
        persistent_cd : bool = True):
        """Train

        Args:
            epoch_idx (int): Epoch idx, only for logging
            all_visible_samples (np.array): All visible samples in shape (no_samples, no_visible)
            batch_size (int): Batch size
            no_cd_steps (int): No CD steps
            no_iter (int, optional): No iterations to run. Defaults to 1000.
            learning_rate_weights (float, optional): Learning rate for weights. Defaults to 0.1.
            learning_rate_biases (float, optional): Learning rate for biases, typically 0.1 to 0.01 times the weights. Defaults to 0.01.
            persistent_cd (bool, optional): True for persistent CD. Defaults to True.
        """

        # Set learning rate
        self.lr_weights.assign(learning_rate_weights)
        self.lr_biases.assign(learning_rate_biases)

        no_training_samples = all_visible_samples.shape[0]

        gibbs = GibbsSampler()

        # Loop
        for iteration in range(0,no_iter):

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
                hidden_batch_awake=hidden_batch_awake,
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
            print("Epoch: %d Iteration: %d / %d Change weights: %.16f bias visible: %.16f bias hidden: %.16f" % (
                epoch_idx,
                iteration, 
                no_iter,
                change_weights,
                change_bias_visible,
                change_bias_hidden))

    def sample_model_dist(self, visible_batch : np.array, no_cd_steps : int) -> Tuple[np.array,np.array]:
        """Dream to generate new samples

        Args:
            visible_batch (np.array): Visible batch to start in shape (batch_size, no_visible)
            no_cd_steps (int): No CD steps to run

        Returns:
            Tuple[np.array,np.array]: Visible and hidden batch after sampling, in sizes (batch_size, no_visible) and (batch_size, no_hidden) 
        """
        batch_size = visible_batch.shape[0]
        hidden_batch = np.zeros((batch_size,self.no_hidden))
        
        gibbs = GibbsSampler()

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

        return (visible_batch, hidden_batch)