# Delivery-Duration-Prediction
Predicting how long it takes for an order to be delivered — before the rider even hits the road.


## Problem Statement

Accurately predicting delivery time is key to keeping customers happy and running operations smoothly. This project aims to build a machine learning model that estimates delivery duration (in minutes) based on different store and order details.


## Data Collection
The dataset for this project comes from Stratascratch, sourced from one of their data science projects. It contains around 197,000 rows and 16 columns, capturing various store and order details to help predict delivery duration.


# Key Steps
### Data Preprocessing
- Handling Outliers: Applied log transformation on the target variable to manage extreme delivery times without removing potentially valid records.

- Missing Value Treatment: Imputation strategies such as  (mean, median, mode) and group based imputation are used for imputing nans in numerical/categorical columns.

- Categorical Encoding: Encoded categorical columns using One-Hot Encoding.

- Feature Engineering: Created new features by using existing ones to minimize noise and enhancing the model’s ability to learn important patterns.

- Feature Correlation Analysis: Analyzed correlation among numerical features to understand dependencies and identify multicollinearity.

- Feature Importance: After model training, inspected feature importance to figure out which features contributed the most to predictions, helping with potential feature selection and model refinement.
## Tools and Libraries
- Python

- Pandas, NumPy

- Scikit-Learn

-  Keras
## Model Selection

I experimented with several machine learning models:

| Model                     | Feature Set | RMSE (minutes) |
|----------------------------|-------------|----------------|
| Decision Tree              | Full Data   | 260.77         |
| Random Forest              | Full Data   | 212.92         |
| LightGBM (LGBM)            | Full Data   | 84.32          |
| Multi-Layer Perceptron (MLP) | Full Data | **0.18**        |

After evaluating the results, the **Multi-Layer Perceptron (MLP)** was selected due to its significantly better performance compared to tree-based models.

### Final Model Details
- **Architecture:** Two hidden layers
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Evaluation Metric:** Root Mean Squared Error (RMSE)

### Why RMSE?
I chose RMSE because it reports errors in the same unit as the target variable (minutes), making it easy to interpret how far off the model is in real-world terms.  


### Final Performance
- **Test RMSE:** ~17 minutes  
(Meaning the model’s predictions are off by approximately 17 minutes on average.)
## Future Work
For future improvements, I plan to enhance the encoding of categorical variables, especially the store_primary_category column, by grouping similar categories together, which will help reduce noise from one-hot encoding. Additionally, I intend to further optimize the neural network by experimenting with different configurations of layers, neurons, activation functions, and optimizers to boost performance.
