import warnings

warnings.filterwarnings('ignore')

import os
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from utils import evaluate_ml, extract

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)


def load_image(data_dir):
    print("Loading all images")
    # Get the list of subdirectories (i.e., class names)
    class_names = os.listdir(data_dir)
    # Initialize the arrays to store the images and class labels
    X = []
    y = []
    # Loop over each class and load the images
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))  # Resize the image
            X.append(img)
            y.append(class_name)

    return shuffle(np.array(X), np.array(y))


# Set the path to the directory containing the image folders
data_dir = 'D:/Abdur/Woodchip/batch_2_classes/data'  # 'D:\Abdur\Woodchip\enviva_whole'
data_name = 'Drax'
feature_extract = False  # set this to true if you want to extract the features from image.


# Assuming X and y are numpy arrays or lists
def shuffle_data(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


if feature_extract:
    # load image from directory
    X, y = load_image(data_dir)
    X, y = shuffle_data(X, y)

    print(X.shape)
    print(y.shape)

    ## Define the K-fold cross-validation splitter
    kfold = KFold(n_splits=4, shuffle=True, random_state=seed)
    fold = 1

    for train_idx, val_idx in kfold.split(X):
        print("Started Fold: ", fold, 'Feature Extraction')
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        # Extract features
        extract(X_train, y_train, fold, data_name, split='train')
        extract(X_val, y_val, fold, data_name, split='test')

        train = pd.read_csv(f'data/{data_name}_{fold}_train.csv')
        test = pd.read_csv(f'data/{data_name}_{fold}_test.csv')

        evaluate_ml(train, test, data_name, fold)
        fold = fold + 1

else:
    for fold in [1, 2, 3, 4]:
        train = pd.read_csv(f'data/{data_name}_{fold}_train.csv')
        test = pd.read_csv(f'data/{data_name}_{fold}_test.csv')

        evaluate_ml(train, test, data_name, fold)
        fold = fold + 1
