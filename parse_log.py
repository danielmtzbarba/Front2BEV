import pandas as pd
import re

# Load the log file
file_path = "analysis.log"

with open(file_path, "r") as file:
    log_contents = file.readlines()

# Initialize dictionary to store experiment data
experiment_data = {}

# Regular expression to match experiment and run lines
experiment_regex = r"EXPERIMENT: (.*?) - DATASET"
run_regex = r"Run (\d+) - test mIoU: ([0-9.]+)"

# Loop through the log to extract experiment and run data
for line in log_contents:
    experiment_match = re.search(experiment_regex, line)
    if experiment_match:
        current_experiment = experiment_match.group(1)
        if current_experiment not in experiment_data:
            experiment_data[current_experiment] = []

    run_match = re.search(run_regex, line)
    if run_match:
        run_number = int(run_match.group(1))
        miou_value = float(run_match.group(2))

        # Append the mIoU value to the current experiment's list
        experiment_data[current_experiment].append(miou_value)

# Create a DataFrame where each column is an experiment, and rows are the runs
df_experiments = pd.DataFrame.from_dict(experiment_data, orient="index").transpose()

# Save the DataFrame to a CSV file
csv_file_path = "experiment_results.csv"
df_experiments.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")
