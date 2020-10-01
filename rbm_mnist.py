import numpy as np
from rbm import RBM
import tensorflow as tf
from matplotlib import pyplot as plt

from pathlib import Path

import sys
from enum import Enum

class Mode(Enum):
    TRAIN_AND_SAVE = 0
    LOAD_AND_MAKE_FIGURES = 1

def print_help_and_exit():
    print("Usage:")
    print("python rbm_mnist.py --train-and-save")
    print("or")
    print("python rbm_mnist.py --load-and-make-figures")
    sys.exit(0)

def save_fig(figures_dir, fname):
    plt.savefig(figures_dir + fname + ".png", dpi=200)

if __name__ == "__main__":

    # Mode
    if len(sys.argv) != 2:
        print_help_and_exit()

    # Arg
    if sys.argv[1] == "--train-and-save":
        mode = Mode.TRAIN_AND_SAVE
    elif sys.argv[1] == "--load-and-make-figures":
        mode = Mode.LOAD_AND_MAKE_FIGURES
    else:
        print_help_and_exit()

    # Load mnist
    df = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
        )

    all_visible_samples_mat = df[0][0] / 256.0
    no_samples = len(all_visible_samples_mat)

    # Flatten the 28x28 pictures into 28*28=784 vector
    x = all_visible_samples_mat.shape[1]
    y = all_visible_samples_mat.shape[2]
    all_visible_samples = np.zeros((no_samples,x*y))
    for i in range(0,no_samples):
        all_visible_samples[i] = all_visible_samples_mat[i].flatten()

    # Equal number of visible and hidden units
    no_visible = all_visible_samples.shape[1]
    no_hidden = no_visible

    # RBM
    rbm = RBM(
        no_visible=no_visible,    
        no_hidden=no_hidden,
        std_dev_weights=0.001,
        std_dev_biases=0.00001
        )

    # Prepare to save
    ckpt = tf.train.Checkpoint(
        weights=rbm.weights, 
        bias_visible=rbm.bias_visible,
        bias_hidden=rbm.bias_hidden
        )

    no_epochs = 20
    manager = tf.train.CheckpointManager(ckpt, './rbm_mnist.ckpt', max_to_keep=no_epochs)
    ckpt.restore(manager.checkpoints[-1])

    if mode == Mode.TRAIN_AND_SAVE:

        # Save initial!
        manager.save()

        no_iter_per_epoch = 500
        for i_epoch in range(0,no_epochs):
            print("--------------- %d / %d ---------------" % (i_epoch,no_epochs))

            # Train
            rbm.train(
                all_visible_samples=all_visible_samples,
                batch_size=5,
                no_cd_steps=1,
                no_iter=no_iter_per_epoch,
                learning_rate_weights=0.00001,
                learning_rate_biases=0.0000001,
                persistent_cd=True
            )

            # Save!
            manager.save()

    else:

        # Load!

        figures_dir = "figures/"
        Path(figures_dir).mkdir(parents=True, exist_ok=True)

        # Load weights to show convergence
        idxs = np.random.randint(low=0,high=x*y,size=(20,2))
        idxs = [tuple(idx) for idx in idxs] 

        # Initial
        weights_ex = {}
        for idx in idxs:
            weights_ex[idx] = []

        # Weights at each epoch
        for i_epoch in reversed(range(0,no_epochs)):
            ckpt.restore(manager.checkpoints[-i_epoch-1])
            for idx in idxs:
                weights_ex[idx].append(rbm.weights.numpy()[idx[0],idx[1]])

        print(weights_ex)
        plt.figure()
        for idx in idxs:
            plt.plot(weights_ex[idx])
        plt.title("Convergence of weights")
        plt.xlabel("Epoch")
        save_fig(figures_dir,"convergence_weights")

        # Load final
        ckpt.restore(manager.latest_checkpoint)

        idxs = list(range(1,10))
        for idx in idxs:
            plt.matshow(all_visible_samples_mat[idx])
            plt.title("Training data %03d" % idx)
            save_fig(figures_dir,"training_data_%03d" % idx)
            plt.close()

        # Histogram weights
        plt.figure()
        plt.hist(rbm.weights.numpy().flatten())
        plt.title("Histogram weights")
        save_fig(figures_dir,"hist_weights")

        plt.figure()
        plt.hist(rbm.bias_visible.numpy().flatten())
        plt.title("Histogram bias visible")
        save_fig(figures_dir,"hist_bias_visible")

        plt.figure()
        plt.hist(rbm.bias_hidden.numpy().flatten())
        plt.title("Histogram bias hidden")
        save_fig(figures_dir,"hist_bias_hidden")

        # Testing data
        testing_mat = df[1][0] / 256.0
        no_testing = len(testing_mat)
        testing = np.zeros((no_testing,x*y))
        for i in range(0,no_testing):
            testing[i] = testing_mat[i].flatten()

        # Sleep phase sample
        idxs = list(range(40,50))
        visible_batch = testing[idxs]
        visible_sampled, hidden_sampled = rbm.sample_model_dist(
            visible_batch=visible_batch, 
            no_cd_steps=20
            )

        for i in range(0,len(idxs)):
            idx = idxs[i]

            mat = np.reshape(visible_batch[i], newshape=(x,y))
            plt.matshow(mat)
            plt.title("Dreaming %03d starting visible" % idx)
            save_fig(figures_dir,"dreaming_%03d_starting" % idx)
            if i != len(idxs) - 1:
                plt.close()

            mat = np.reshape(visible_sampled[i], newshape=(x,y))
            plt.matshow(mat)
            plt.title("Dreaming %03d sampled visible" % idx)
            save_fig(figures_dir,"dreaming_%03d_visible" % idx)
            if i != len(idxs) - 1:
                plt.close()

            mat = np.reshape(hidden_sampled[i], newshape=(x,y))
            plt.matshow(mat)
            plt.title("Dreaming %03d sampled hidden" % idx)
            save_fig(figures_dir,"dreaming_%03d_hidden" % idx)
            if i != len(idxs) - 1:
                plt.close()

        plt.show()

