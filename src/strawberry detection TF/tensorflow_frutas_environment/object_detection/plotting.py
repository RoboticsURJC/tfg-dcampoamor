import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import Sequential


def plot_loss(history, output_dir):
    loss = history.history['loss']
    steps = np.arange(len(loss))
    plt.plot(steps, loss, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, 'loss.png'))

def plot_accuracy(history, output_dir):
    accuracy = history.history['accuracy']
    steps = np.arange(len(accuracy))
    plt.plot(steps, accuracy, label='Training Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))



