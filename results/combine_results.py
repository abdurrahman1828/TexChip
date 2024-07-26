import pandas as pd
import glob

# Define the path pattern for the CSV files
csv_pattern = "EnvivaWhole_Fold_*.csv"

# Load all CSV files
all_files = glob.glob(csv_pattern)
print(all_files)
dfs = []

for file in all_files:
    df = pd.read_csv(file, header=0)
    print(df)
    dfs.append(df)

# Concatenate all dataframes row-wise
combined_df = pd.concat(dfs, ignore_index=True)
print(combined_df)
# Assign column names
combined_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]

# Group by Model to calculate mean and standard deviation
mean_df = combined_df.groupby("Model").mean()
std_df = combined_df.groupby("Model").std()

# Combine mean and standard deviation into a single dataframe
summary_df = pd.DataFrame({
    "Model": mean_df.index,
    "Mean_Accuracy": mean_df["Accuracy"],
    "Std_Accuracy": std_df["Accuracy"],
    "Mean_Precision": mean_df["Precision"],
    "Std_Precision": std_df["Precision"],
    "Mean_Recall": mean_df["Recall"],
    "Std_Recall": std_df["Recall"],
    "Mean_F1-Score": mean_df["F1-Score"],
    "Std_F1-Score": std_df["F1-Score"],
}).reset_index(drop=True)

# Save the summary dataframe to a new CSV file
summary_df.to_csv("Enviva_summary.csv", index=False)

print("Summary saved")
