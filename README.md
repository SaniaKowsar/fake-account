# Fake Account Detection in Social Media

## Overview
This project uses machine learning to detect fake accounts on social media platforms based on various features like number of followers, posts, and profile completeness.

## Features
- Data Preprocessing and Feature Engineering
- ML Models: Logistic Regression, Random Forest, SVM
- Evaluation: Accuracy, Precision, Recall, F1 Score
- Model Persistence with Joblib
- Easy predictions using a trained model

## How to Run

1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:  
   ```bash
   python train_model.py
   ```

3. Predict using the model:  
   ```bash
   python predict.py
   ```

## Files
- `train_model.py`: Code to train and save the ML model
- `predict.py`: Use trained model to predict fake/genuine accounts
- `data.csv`: Sample dataset
- `model.pkl`: Saved ML model

## Requirements
Python 3.7+

## Future Work
- Add Streamlit UI
- Use real-time data from Twitter API
- Deploy on cloud
