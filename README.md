# Credit Card Fraud Detection Using Machine Learning

## Project Overview

Credit card fraud is a major challenge for banks and financial institutions. Fraudulent transactions can cause significant financial losses and reduce customer trust.

This project uses Machine Learning to identify whether a credit card transaction is fraudulent or genuine based on transaction data.

---

## Objective

The main objective of this project is to build a machine learning model that can accurately classify transactions as:

- Genuine Transaction (Class = 0)
- Fraudulent Transaction (Class = 1)

---

## Dataset

Dataset: Credit Card Fraud Detection Dataset

Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset contains:

- Transaction features
- Transaction amount
- Fraud labels
- Genuine labels

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- VS Code

---

## Libraries Used

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report