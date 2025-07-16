import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create structured data for fake accounts
fake_data = pd.DataFrame({
    'account_age_days': np.random.randint(1, 30, 250),
    'num_followers': np.random.randint(0, 50, 250),
    'num_following': np.random.randint(1000, 5000, 250),
    'num_posts': np.random.randint(0, 10, 250),
    'profile_picture': 0,
    'bio_filled': 0,
    'is_fake': 1
})

# Create structured data for real accounts
real_data = pd.DataFrame({
    'account_age_days': np.random.randint(300, 3000, 250),
    'num_followers': np.random.randint(500, 10000, 250),
    'num_following': np.random.randint(50, 500, 250),
    'num_posts': np.random.randint(100, 2000, 250),
    'profile_picture': 1,
    'bio_filled': 1,
    'is_fake': 0
})

# Combine data
df = pd.concat([fake_data, real_data], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and labels
X = df.drop('is_fake', axis=1)
y = df['is_fake']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "fake_account_model.pkl")

print("✅ Better model trained and saved at:", os.path.abspath("fake_account_model.pkl"))
