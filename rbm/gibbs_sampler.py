import numpy as np
import tensorflow as tf
import sys
np.seterr(all='raise')

from typing import Any, Tuple

class GibbsSampler:

    def __init__(self):
        pass

    def _get_prob(self, ef_0 : np.array, ef_1 : np.array) -> np.array:
        """Get probability from error function

        Args:
            ef_0 (np.array): Error function for units = 0
            ef_1 (np.array): Error function for units = 1

        Returns:
            np.array: exp(ef_1) / (exp(ef_0) + exp(ef_1)) constrained in [0,1]
        """

        ef_1[np.where(ef_1 < -10.0)] = -10.0
        ef_1[np.where(ef_1 > 10.0)] = 10.0

        try:
            prob = np.exp(ef_1) / (np.exp(ef_0) + np.exp(ef_1))
        except Exception as w:
            print("Could not calculate prob: " + str(w))
            print(ef_1)
            print(ef_0)
            print("Quitting")
            sys.exit(0)

        # Constrain in [0,1]
        prob[np.where(prob < 0)] = 0.0
        prob[np.where(prob > 1)] = 1.0

        return prob

    def activation_hidden(self, 
        visible_batch : np.array,
        bias_hidden : np.array,
        weights : np.array,
        i_batch : int
        ) -> Tuple[np.array, np.array]:
        """Activation function for hidden units from visible

        Args:
            visible_batch (np.array): Batch of visible data in shape (batch_size, no_visible)
            bias_hidden (np.array): Visible bias values in shape (no_hidden, 1)
            weights (np.array): Weights in shape (no_visible, no_hidden)
            i_batch (int): Batch index

        Returns:
            Tuple[float,float]: (Error function for 0 (=0), Error function for 1)
        """
    
        # Energy func for unit = 0
        ef_0 = np.full(shape=len(bias_hidden),fill_value=0.0) # trivial

        # Energy func for unit = 1
        ef_1 = bias_hidden[:,0] + np.dot(np.transpose(weights), visible_batch[i_batch])
        
        return (ef_0, ef_1)

    def activation_visible(self, 
        hidden_batch : np.array,
        bias_visible : np.array,
        weights : np.array,
        i_batch : int
        ) -> Tuple[float,float]:
        """Activation function for visible units from hidden

        Args:
            hidden_batch (np.array): Batch of hidden data in shape (batch_size, no_hidden)
            bias_visible (np.array): Visible bias values in shape (no_visible, 1)
            weights (np.array): Weights in shape (no_visible, no_hidden)
            i_batch (int): Batch index

        Returns:
            Tuple[float,float]: (Error function for 0 (=0), Error function for 1)
        """
        
        # Energy func for unit = 0
        ef_0 = np.full(shape=len(bias_visible),fill_value=0.0) # trivial

        # Energy func for unit = 1
        ef_1 = bias_visible[:,0] + np.dot(weights, hidden_batch[i_batch])

        return (ef_0, ef_1)

    def _sample_x_given_y(self,
        y_batch : np.array,
        no_x : int,
        ef : Any, 
        binary : bool = True) -> np.array:
        """Sample x given y

        Args:
            y_batch (np.array): Batch of y data in size (batch_size, y_data_size)
            no_x (int): x data size
            ef (Any): Error function; should take only one argument = batch index
            binary (bool, optional): True to binarize the output, False for raw probabilities. Defaults to True.

        Returns:
            np.array: Sampled data in size (batch_size, x_data_size = no_x)
        """
        
        # Get batch size
        batch_size = y_batch.shape[0]
        
        # Hidden batch
        x_batch = np.zeros((batch_size, no_x))

        # Sample
        for i_batch in range(0,batch_size):
            
            # Energy function
            [ef_0, ef_1] = ef(i_batch)

            # Prob
            prob = self._get_prob(ef_0,ef_1)
        
            if binary:
                # Sample
                r = np.random.rand(len(prob))
                x_batch[i_batch][r < prob] = 1
            else:
                # Store probability
                x_batch[i_batch] = prob

        # Fix type
        if binary:
            x_batch = x_batch.astype('int')

        return x_batch

    def sample_hidden_given_visible(self, 
        visible_batch : np.array, 
        bias_hidden_tf : tf.Variable,
        weights_tf : tf.Variable, 
        binary : bool = True) -> np.array:
        """Sample hidden units from the visible

        Args:
            visible_batch (np.array): Batch of visible data in shape (batch_size, no_visible)
            bias_hidden_tf (tf.Variable): Tensorflow variable for hidden bias
            weights_tf (tf.Variable): Tensorflow variable for weights
            binary (bool, optional): True to binarize, False for raw probabilities. Defaults to True.

        Returns:
            np.array: Samples in shape (batch_size, no_hidden)
        """

        bias_hidden = np.transpose(bias_hidden_tf.numpy())
        weights = weights_tf.numpy()

        ef = lambda i_batch: self.activation_hidden(
            visible_batch=visible_batch,
            bias_hidden=bias_hidden,
            weights=weights,
            i_batch=i_batch
            )
        
        no_hidden = bias_hidden_tf.shape[1]

        return self._sample_x_given_y(
            y_batch=visible_batch,
            no_x=no_hidden,
            ef=ef,
            binary=binary
            )

    def sample_visible_given_hidden(self,
        hidden_batch : np.array, 
        bias_visible_tf : tf.Variable,
        weights_tf : tf.Variable, 
        binary : bool = True) -> np.array:
        """Sample visible units from the hidden

        Args:
            hidden_batch (np.array): Batch of hidden data in shape (batch_size, no_hidden)
            bias_visible_tf (tf.Variable): Tensorflow variable for visible bias
            weights_tf (tf.Variable): Tensorflow variable for weights
            binary (bool, optional): True to binarize, False for raw probabilities. Defaults to True.

        Returns:
            np.array: Samples in shape (batch_size, no_visible)
        """

        bias_visible = np.transpose(bias_visible_tf.numpy())
        weights = weights_tf.numpy()

        ef = lambda i_batch: self.activation_visible(
            hidden_batch=hidden_batch,
            bias_visible=bias_visible,
            weights=weights,
            i_batch=i_batch
            )
        
        no_visible = bias_visible_tf.shape[1]

        return self._sample_x_given_y(
            y_batch=hidden_batch,
            no_x=no_visible,
            ef=ef,
            binary=binary
            )

