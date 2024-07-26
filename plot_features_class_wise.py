import os
import cv2
import mahotas.features.texture as texture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Function to compute texture features
def compute_texture_features(img):
    features = texture.haralick(img, return_mean_ptp=True, compute_14th_feature=True)
    return features

# Function to load images and compute texture features
def load_and_compute_features(data_dir):
    print("Loading and computing texture features for all images")
    # Get the list of subdirectories (i.e., class names)
    class_names = os.listdir(data_dir)
    # Initialize lists to store computed features and class labels
    features_list = []
    class_labels = []
    # Loop over each class and load the images
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            # Compute texture features
            features = compute_texture_features(img)
            features_list.append(features)
            class_labels.append(class_name)

    return np.array(features_list), np.array(class_labels)

# Function to plot box plots for each feature separately
def plot_feature_boxplots(features, class_labels, feature_names):
    num_classes = np.unique(class_labels)
    num_features = len(feature_names)

    for i, feature_name in enumerate(feature_names):
        fig, ax = plt.subplots(figsize=(4, 3))
        data = [features[class_labels == cls][:, i] for cls in num_classes]
        ax.boxplot(data, labels=num_classes)
        ax.set_title(feature_name)
        ax.set_ylabel("Feature Value")
        ax.set_xlabel("Class")
        plt.tight_layout()
        plt.savefig(f"plots/{feature_name}_box_final_EnvivaWhole.jpg", dpi = 600, bbox_inches = 'tight')
        plt.show()

# Define feature names
feature_names = ['ASM', 'Contrast', 'Correlation', 'SSV', 'IDM', 'SA', 'SV', 'SE', 'Entropy', 'DV',
                                   'DE', 'IMC1', 'IMC2', 'MCC',
                                   'DfASM', 'DfContrast', 'DfCorrelation', 'DfSSV', 'DfIDM', 'DfSA', 'DfSV', 'DfSE',
                                   'DfEntropy', 'DfDV', 'DfDE', 'DfIMC1', 'DfIMC2', 'DfMCC']

# Load images and compute features
data_dir =  'D:\Abdur\Woodchip\enviva_whole' #'D:/Abdur/Woodchip/batch_2_classes/data'
features, class_labels = load_and_compute_features(data_dir)

# Plot box plots for each feature separately
plot_feature_boxplots(features, class_labels, feature_names)