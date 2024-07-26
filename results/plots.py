import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

plt.style.use('classic')

# Define the path pattern for the CSV files
csv_pattern = "EnvivaWhole_Fold_*.csv"

# Load all CSV files and combine them
all_files = glob.glob(csv_pattern)
combined_df = pd.concat([pd.read_csv(file, header=0) for file in all_files], ignore_index=True)
combined_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]

# Define the models
models = combined_df["Model"].unique()

# Create a dictionary to store scores for each metric for each model
model_scores = {
    model: combined_df[combined_df["Model"] == model][["Accuracy", "Precision", "Recall", "F1-Score"]].values.tolist()
    for model in models}

# Define metrics
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

# Extract the first two significant letters in uppercase for each model name
model_labels = [''.join(filter(str.isupper, model))[:3] for model in models]

# Plotting for each metric separately
for metric in metrics:
    # Create a new figure for each metric
    plt.figure(figsize=(5, 3))
    #plt.title(metric)
    plt.xlabel("Model")
    plt.ylabel(f"{metric}")
    plt.xticks(rotation=45)

    # Extract scores for the current metric
    scores = [np.array(model_scores[model])[:, metrics.index(metric)] for model in models]

    # Plot boxplot for the current metric
    plt.boxplot(scores, labels=model_labels)

    # Save the figure with a descriptive filename
    plt.savefig(f"{metric}_boxplot_EnvivaWhole.jpg", dpi = 600, bbox_inches = 'tight')

    # Show the plot
    plt.tight_layout()
    plt.show()
