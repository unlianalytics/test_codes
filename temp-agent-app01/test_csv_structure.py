import pandas as pd

# Test the CSV structure
try:
    print("=== Testing LTE CSV ===")
    lte_df = pd.read_csv('csv_files/Parameters_LTE.csv', dtype=str, encoding='latin1')
    print(f"LTE CSV columns: {list(lte_df.columns)}")
    print(f"LTE CSV shape: {lte_df.shape}")
    print(f"First few rows of 'Abbreviated name' column:")
    if 'Abbreviated name' in lte_df.columns:
        print(lte_df['Abbreviated name'].head())
    else:
        print("'Abbreviated name' column not found!")
        print("Available columns:", list(lte_df.columns))
    
    print("\n=== Testing NR CSV ===")
    nr_df = pd.read_csv('csv_files/Parameters_NR.csv', dtype=str, encoding='latin1')
    print(f"NR CSV columns: {list(nr_df.columns)}")
    print(f"NR CSV shape: {nr_df.shape}")
    print(f"First few rows of 'Abbreviated name' column:")
    if 'Abbreviated name' in nr_df.columns:
        print(nr_df['Abbreviated name'].head())
    else:
        print("'Abbreviated name' column not found!")
        print("Available columns:", list(nr_df.columns))
        
except Exception as e:
    print(f"Error: {e}") 