import pandas as pd
import scipy.stats as stats

# Load your CSV file
df = pd.read_csv("experiment_results.csv")

# Select the two columns you want to compare (replace with actual column names)
col1 = "f2b-mini-ved-rgb"
col2 = "f2b-mini-rgved-rgbd"

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(df[col1].dropna(), df[col2].dropna())

# Print the results
print(f"ANOVA results:\nF-statistic: {f_stat}\nP-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("There is a statistically significant difference between the two groups.")
else:
    print("There is no statistically significant difference between the two groups.")
