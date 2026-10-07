# 🏠 San Francisco Rent Price Prediction

A machine learning and data analysis project that predicts monthly apartment rental prices in San Francisco using property characteristics from Craigslist housing listings.

The project combines a **Jupyter Notebook for Exploratory Data Analysis (EDA), feature engineering, and regression model comparison** with an **interactive Streamlit application** where users can enter apartment features and receive an estimated monthly rent.

## 📌 Project Overview

The main goal of this project is to understand the factors that influence San Francisco rental prices and build a machine learning model capable of estimating apartment rent from property features.

The project analyzes variables such as:

- Apartment size
- Number of bedrooms
- Number of bathrooms
- Laundry availability
- Parking type
- Pet policy
- Housing type
- Neighborhood district

The analysis explores questions such as:

- How strongly does apartment size affect rent?
- Do additional bedrooms and bathrooms increase rental prices?
- How important are parking and laundry amenities?
- Which regression algorithm performs best?
- How accurately can monthly rent be estimated from listing features?

## 📊 Dataset

The project uses:

```text
sf_clean.csv
```

The data comes from cleaned **San Francisco Craigslist apartment listings from October 2020**.

The notebook loads **989 listings with 9 features**:

| Column | Description |
|---|---|
| `price` | Monthly rental price in dollars |
| `sqft` | Apartment size in square feet |
| `beds` | Number of bedrooms |
| `bath` | Number of bathrooms |
| `laundry` | Laundry facility type |
| `pets` | Pet policy |
| `housing_type` | Type of housing |
| `parking` | Parking availability/type |
| `hood_district` | Neighborhood district code |

The notebook reports **no missing values** in the dataset.

## 📈 Dataset Highlights

Some descriptive statistics from the notebook include:

| Metric | Average |
|---|---:|
| Monthly Rent | **$3,595** |
| Apartment Size | **977 sqft** |
| Bedrooms | **1.68** |
| Bathrooms | **1.39** |

The median monthly rent in the original dataset is approximately:

```text
$3,300
```

The observed rental prices range from approximately **$750 to $19,000** before outlier filtering.

## 🔗 Correlation Analysis

The notebook examines relationships between numerical features and rental price.

Approximate correlations with `price` are:

| Feature | Correlation with Price |
|---|---:|
| `sqft` | **0.836** |
| `bath` | **0.691** |
| `beds` | **0.673** |
| `hood_district` | **0.013** |

Apartment size has the strongest direct numerical relationship with rental price in the dataset.

## 🔎 Exploratory Data Analysis

The Jupyter Notebook (`SanFranciscoHousing.ipynb`) includes:

- Dataset inspection
- Shape and data-type analysis
- Missing-value checks
- Descriptive statistics
- Correlation analysis
- Correlation heatmap
- Price correlation ranking
- Housing-price distribution
- Price outlier analysis
- Bedroom distribution
- Bathroom distribution
- Laundry distribution
- Parking distribution
- Pet-policy distribution
- Housing-type distribution
- Square-footage outlier analysis
- Neighborhood-district analysis
- Regression plots between major features and price

## 🧹 Feature Engineering

The notebook performs several preprocessing and feature-engineering steps before model training.

### Laundry Encoding

Laundry options are converted into ordinal numeric values:

```text
No laundry → 0
On-site    → 1
In-unit    → 2
```

### Parking Encoding

Parking options are converted into numeric values:

```text
No parking → 0
Off-street → 1
Protected  → 2
Valet      → 3
```

### Categorical Cleaning

Prefix labels such as `(a)`, `(b)`, and `(c)` are removed from:

- Pet policy
- Housing type

### Outlier Filtering

Extreme rental prices are removed using the **97th percentile** of the price distribution.

### Feature Transformation

The notebook applies squared transformations to:

```text
beds
bath
sqft
```

Categorical variables are converted to dummy variables using `pandas.get_dummies()`.

The resulting features are scaled using `MinMaxScaler`.

## 🤖 Machine Learning Models

The notebook compares several regression algorithms:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Extra Tree Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor
- XGBoost Regressor

The data is divided into training and testing sets using an **80/20 split** with `random_state=42`.

## 🏆 Model Performance

The notebook reports the following results:

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| **Gradient Boosting Regressor** | **0.801** | **$524** | **$398** |
| XGBoost Regressor | 0.767 | $567 | $437 |
| Linear Regression | 0.721 | $619 | $486 |
| Lasso | 0.719 | $622 | $488 |
| Ridge | 0.708 | $634 | $501 |
| K-Nearest Neighbors | 0.580 | $760 | $579 |
| Extra Tree Regressor | 0.523 | $810 | $617 |
| ElasticNet | 0.129 | $1,095 | $856 |

The **Gradient Boosting Regressor** achieved the best overall performance in the notebook.

### Best Model

```text
GradientBoostingRegressor
R²   ≈ 0.801
RMSE ≈ $524
MAE  ≈ $398
```

An R² score of approximately **0.80** means the model explains a substantial portion of the variation in rental prices within this dataset.

## 💡 Main Findings

The notebook concludes that the Gradient Boosting model provides the strongest rental-price predictions among the tested algorithms.

The analysis also identifies important property characteristics associated with rental value, particularly:

- Square footage
- Number of bathrooms
- Number of bedrooms
- Parking availability

Square footage shows the strongest simple numerical correlation with price.

## 🖥️ Streamlit Rent Predictor

The project also includes an interactive Streamlit application (`sanfrancisco.py`).

The application trains a **GradientBoostingRegressor** and allows users to estimate monthly rent by selecting apartment characteristics.

### Prediction Inputs

Users can enter:

- Apartment size in square feet
- Number of bedrooms
- Number of bathrooms
- Housing type
- Laundry type
- Parking type
- Pet policy
- Neighborhood district

After clicking **Predict rent**, the application returns an estimated monthly rental price.

## 💵 Prediction Output

The dashboard displays:

- Estimated monthly rent
- Typical prediction error based on RMSE
- Median rent for listings with the same number of bedrooms

Example output structure:

```text
Estimated monthly rent: $X,XXX
Typical model error: ± $XXX
```

## 📊 Dashboard Sections

The Streamlit application contains three tabs.

### 1. Predict

Users configure the apartment characteristics and receive a real-time rent estimate from the trained Gradient Boosting model.

### 2. Data Insights

The dashboard includes interactive Plotly visualizations for:

- Rent distribution
- Apartment size vs. rent
- Average rent by number of bedrooms
- Rent by neighborhood district
- Correlation matrix

These charts help users understand the main patterns in the housing dataset.

### 3. Model

The model section displays:

- R² score
- RMSE
- MAE
- Actual vs. predicted rent scatter plot
- Feature importance

The actual-vs-predicted chart includes a reference line representing perfect predictions.

## ⚙️ Streamlit Modeling Pipeline

The Streamlit application uses a slightly simplified modeling pipeline compared with the notebook.

It:

1. Removes extreme prices above the 97th percentile
2. Converts categorical features using one-hot encoding
3. Splits the data into training and test sets
4. Trains a `GradientBoostingRegressor`
5. Calculates R², RMSE, and MAE
6. Uses the trained model for interactive rent predictions

This allows the original notebook analysis to be turned into a practical prediction application.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Scikit-learn
- XGBoost
- Gradient Boosting
- Regression Modeling
- Jupyter Notebook

## 🎯 Project Purpose

This project demonstrates practical skills in:

- Exploratory Data Analysis
- Data cleaning
- Feature engineering
- Categorical encoding
- Outlier detection
- Regression analysis
- Machine learning model comparison
- Gradient Boosting
- XGBoost
- Model evaluation
- Feature importance analysis
- Interactive prediction applications
- Streamlit dashboard development
---

Built with Python, Scikit-learn, and Streamlit to analyze and predict San Francisco apartment rental prices. 🏠📊🤖
