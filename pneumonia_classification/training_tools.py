import sys
import os
import glob
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import TensorBoard

from keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import Input, Model
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import cv2
from pathlib import Path
import datetime
from typing import Tuple
import copy
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
# sns.set_style('darkgrid')

def evaluate_model(model, x_train, x_valid, x_test, y_train, y_valid, y_test, datagen, epochs=40, batch_size=32, forced_training=False, name_suffix="", labels=None):

    print(f"=== MODEL EVALUATION =================================================\n")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])
    model.summary()
    
    model_name = model.name + name_suffix

    MODEL_CHECKPOINT = f"./models/{model_name}.ckpt"
    
    # Log dataset sizes to wandb
    wandb.init(project="pneumonia_classification", name=model_name, config={"name_suffix": name_suffix})
    wandb.log({
        "training_samples": len(x_train),
        "validation_samples": len(x_valid),
        "test_samples": len(x_test)
    })

    if not os.path.exists(MODEL_CHECKPOINT) or forced_training:
        print(f"\n--- Model training ---------------------------------------------------\n")
#         wandb.init(project="pneumonia_classification", name=model.name)
        shutil.rmtree(MODEL_CHECKPOINT, ignore_errors=True)
        
        # tensorboard callback
        log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=20),
            keras.callbacks.ModelCheckpoint(
                filepath=MODEL_CHECKPOINT,
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1),
            tensorboard_callback,
            WandbMetricsLogger()
        ]
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            callbacks=callbacks_list,
            validation_data=datagen.flow(x_valid, y_valid),
            verbose=1)
#         wandb.finish()
        print(f"\n--- Training history -------------------------------------------------\n")

        fig, ax = plt.subplots(1, 2, figsize=(16, 4))
        sns.lineplot(data={k: history.history[k] for k in ('loss', 'val_loss')}, ax=ax[0])
        sns.lineplot(data={k: history.history[k] for k in history.history.keys() if k not in ('loss', 'val_loss')}, ax=ax[1])
        plt.show()

    else:
        print(f"\n--- Model is already trainded ... loading ----------------------------\n")

    model.load_weights(MODEL_CHECKPOINT)

    print(f"\n--- Test Predictions and Metrics -------------------------------------\n")

    y_pred = model.predict(x_test, verbose=0)
    
    y_true = np.argmax(y_test, axis=-1)
    y_pred_classes = np.argmax(y_pred, axis=-1)

    # Calculate accuracy
    test_accuracy = accuracy_score(y_true, y_pred_classes)

    plt.figure(figsize=(6, 3))
    heatmap = sns.heatmap(confusion_matrix(np.argmax(y_test, axis=-1),  np.argmax(y_pred, axis=-1)), annot=True, fmt="d", cbar=True)
    if labels:
        heatmap.yaxis.set_ticklabels(labels, rotation=90, ha='right')
        heatmap.xaxis.set_ticklabels(labels, rotation=0, ha='right')
    heatmap.axes.set_ylabel('True label')
    heatmap.axes.set_xlabel('Predicted label')
    plt.show()

    print()
    classification_report_str = classification_report(np.argmax(y_test, axis=-1), np.argmax(y_pred, axis=-1), target_names=labels, zero_division=0)
    print(classification_report_str)
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                        y_true=np.argmax(y_test, axis=-1),
                                                        preds=np.argmax(y_pred, axis=-1),
                                                        class_names=labels),
        "classification_report": classification_report_str,
        "test_accuracy": test_accuracy
    })
    wandb.finish()
    print(f"\n=== MODEL EVALUATION FINISHED ========================================")