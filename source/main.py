"""cifar10 CNN training script

This script provides:
- load_data(): downloads and prepares CIFAR-10
- build_model(): returns a compiled Keras model
- train(): trains the model and saves checkpoints & plots
- evaluate(): evaluates on test set and prints metrics

When run as a script, it trains for a default number of epochs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = SAMPLE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def load_data(mirror=True):
    """Download and prepare CIFAR-10 dataset.
    If mirror=True uses a Google Cloud mirror which is usually faster.
    Returns: (train_images, train_labels), (test_images, test_labels)
    """
    if mirror:
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-batches-py.tar.gz"
        tf.keras.utils.get_file("cifar-10-batches-py", origin=url, untar=True)
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images  = test_images.astype('float32')  / 255.0
    return (train_images, train_labels), (test_images, test_labels)

def build_model(input_shape=(32,32,3), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def plot_history(history, outdir=PLOTS_DIR):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='val_accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / 'accuracy_curve.png')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / 'loss_curve.png')
    plt.close()

def plot_confusion(y_true, y_pred, outdir=PLOTS_DIR):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
    plt.savefig(outdir / 'confusion_matrix.png')
    plt.close()

def train(epochs=10, batch_size=64, save_model=True, model_path=None):
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('cifar10_best.h5', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_test, y_test), callbacks=callbacks)
    plot_history(history)
    # Evaluate and save confusion matrix
    logits = model.predict(x_test)
    preds = np.argmax(logits, axis=1)
    plot_confusion(y_test.flatten(), preds)
    if save_model:
        model.save(model_path or 'cifar10_trained_model')
    # Print classification report
    print('\nClassification report:')
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))
    return model, history

def evaluate(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {acc:.4f}, loss: {loss:.4f}")
    return loss, acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)