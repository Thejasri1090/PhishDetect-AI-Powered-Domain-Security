import pandas as pd
# Load dataset
dataset_path = '/Users/thejasri/Downloads/urldata.csv'  # Replace with your actual path
df = pd.read_csv(dataset_path)

# Display dataset before feature extraction
print("Dataset Before Feature Extraction:")
print(df.head())

# Feature extraction
df['features'] = df['url'].apply(extract_features)
X = np.array(df['features'].tolist())
y = df['label']

# Display dataset after feature extraction
df_after_extraction = pd.DataFrame(X, columns=[
    'url_length', 'domain_length', 'path_length', 'num_path_segments', 'num_queries',
    'num_dots', 'num_hyphens', 'num_digits', 'num_special_chars', 'num_subdomains',
    'has_https', 'has_dot_com'
])
df_after_extraction['label'] = y
print("\nDataset After Feature Extraction:")
print(df_after_extraction.head())
