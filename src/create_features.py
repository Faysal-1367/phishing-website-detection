import pandas as pd
from feature_extraction import extract_features

# Load Dataset
df = pd.read_csv("dataset/phishing_simple (1).csv")

# Generate Features
features = df["URL"].apply(extract_features)

# Convert to DataFrame
feature_df = pd.DataFrame(
    features.tolist(),
    columns=[
        "url_length",
        "domain_length",
        "dot_count",
        "hyphen_count",
        "digit_count",
        "https",
        "has_at",
        "has_ip"
    ]
)

# Add Label
feature_df["label"] = df["label"]

# Save
feature_df.to_csv(
    "dataset/features.csv",
    index=False
)

print(feature_df.head())
print("Features Saved Successfully!")