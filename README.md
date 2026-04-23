# 🚀 Data2Deploy — End-to-End AutoML Platform

Data2Deploy is a full-stack machine learning platform that enables users to upload datasets, perform automated exploratory data analysis (EDA), train and compare multiple models, track experiments, and export production-ready models as containerized applications.

👉 **Live Demo:** <your-render-link>  
👉 **Repository:** <your-github-link>

---

# 🧠 Overview

Data2Deploy simplifies the entire machine learning lifecycle into a unified workflow:

**Upload → Analyze → Train → Compare → Track → Deploy**

It is designed to bridge the gap between experimentation and deployment by integrating data analysis, model training, and production packaging into a single system.

---

# ✨ Key Features

## 📊 Exploratory Data Analysis (EDA)
- Dataset overview (shape, missing values, duplicates)
- Statistical summaries (mean, median, std, quartiles)
- Correlation analysis and heatmaps
- Automated insights (outliers, skewness, patterns)
- Feature engineering recommendations
- Interactive visualizations (Histogram, Scatter, Box, Bar, Heatmap)

---

## 🤖 Automated Machine Learning
- Supports multiple models (XGBoost, LightGBM, CatBoost, Scikit-learn)
- Automatic and manual feature selection
- Hyperparameter tuning with cross-validation
- Configurable training parameters
- Supports both regression and classification tasks

---

## 📈 Model Evaluation & Comparison
- Compare top-K models side by side
- Evaluation metrics:
  - Regression: MAE, RMSE, R²
  - Classification: Accuracy, Precision, Recall, F1-score
- Prediction previews
- Feature importance insights

---

## 📋 Experiment Tracking
- Track all training runs
- View experiment history
- Compare multiple experiments
- Rerun experiments with same configuration
- Delete unwanted runs

---

## 💾 Model Export & Deployment
- Download trained models
- Automatic containerization
- Export as a standalone ML application
- Includes:
  - Model
  - Preprocessor
  - Prediction interface
  - Dockerfile for deployment

---

# 🔄 Workflow

1. Upload dataset (CSV)
2. Preview and validate data
3. Perform EDA using Insight Studio
4. Select target column and features
5. Configure training parameters
6. Train multiple models
7. Compare results and select best model
8. Track experiments
9. Export model as deployable container

---

# 🏗️ Tech Stack

## Backend
- FastAPI
- Python

## Machine Learning
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

## Data Processing
- Pandas
- NumPy

## Visualization
- Plotly
- Matplotlib
- Seaborn

## Experiment Tracking
- MLflow

## Deployment
- Docker
- Render

---

# ⚙️ Installation

```bash
git clone <your-repo-link>
cd Data2Deploy
pip install -r requirements.txt
