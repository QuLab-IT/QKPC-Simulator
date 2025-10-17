import pandas as pd

# File paths
input_file = "Final_no.txt"  # Replace with actual file name
output_file = "filtered_output.txt"  # New file without negative Cp values

# Read the file (assuming whitespace as delimiter)
df = pd.read_csv(input_file, sep=r'\s+', header=None, engine='python')

# Filter out rows where the third column (Cp) is negative
filtered_df = df[df.iloc[:, 2] >= 0]  # Column index 2 (zero-based)

# Save the new filtered file
filtered_df.to_csv(output_file, sep='\t', index=False, header=False)

print(f"Filtered file saved as: {output_file}")
