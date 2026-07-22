# California Property Close Price Prediction

A machine learning project that predicts the final sale price (ClosePrice) of single-family residential properties in California using historical Multiple Listing Service (MLS) data.

## Project Overview

The goal of this project is to develop and compare multiple machine learning models capable of estimating a property's closing price based on its characteristics, such as:

- Living Area
- Number of Bedrooms
- Number of Bathrooms
- Lot Size
- Property Location
- Additional property features

The project follows a structured 12-week development plan, progressing from data exploration to model development, evaluation, and deployment. 

---

## Dataset

The dataset consists of historical residential property transactions from the California Regional Multiple Listing Service (CRMLS).

Only the following property types are included:

- PropertyType = Residential
- PropertySubType = SingleFamilyResidence

The target variable is:

- **ClosePrice** (Final Sale Price)

The project uses multiple months of historical sales data to train and evaluate prediction models.

---

## Project Objectives

- Explore and understand the housing dataset
- Clean and preprocess raw MLS data
- Engineer useful predictive features
- Compare multiple machine learning algorithms
- Evaluate model performance using regression metrics
- Build an accurate California home price prediction model

---

## Project Structure

```
California-Price-Prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_advanced_models.ipynb
│
├── models/
│
├── results/
│
├── app/
│
└── README.md
```

---

## Machine Learning Models

The project compares multiple regression algorithms, including:

- Linear Regression
- Ridge Regression
- LASSO Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost
- Neural Network (MLP)

Each model is evaluated and compared to identify the best-performing approach.

---

## Evaluation Metrics

Model performance is evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Median Absolute Percentage Error (MdAPE)

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Development Timeline

| Week | Tasks |
|------|------|
| 1 | Environment Setup & Dataset Preparation |
| 2 | Exploratory Data Analysis |
| 3 | Data Cleaning & Preprocessing |
| 4 | Baseline Linear Regression |
| 5 | Decision Tree & Random Forest |
| 6 | Feature Engineering |
| 7 | XGBoost & Hyperparameter Tuning |
| 8 | Model Evaluation |
| 9 | Streamlit Prediction App |
| 10 | Documentation |
| 11 | Presentation Preparation |
| 12 | Final Presentation & Project Delivery |

The project follows the official 12-week internship plan.

---

## Future Improvements

- Geographic feature engineering
- School district integration
- Hyperparameter optimization
- Ensemble learning
- Interactive Streamlit web application

---

## Author

**Jessica Nguyen**

Bachelor of Science in Data Science  
University of Central Florida
