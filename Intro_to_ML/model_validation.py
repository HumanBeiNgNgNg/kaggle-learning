import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# STEP 1. Selecting Data for Modeling
melbourne_file_path = 'Intro_to_ML/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

print(melbourne_data.columns)

# Drop rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)

# STEP 2. Selecting The Prediction Target
y = melbourne_data.Price

# STEP 3. Choosing Features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

print(X.describe())
print(X.head())

# STEP 4. Building Your Initial Model
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model using all available data
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())

print("The predictions are:")
print(melbourne_model.predict(X.head()))

# STEP 5. Calculating Mean Absolute Error (MAE)
predicted_home_prices = melbourne_model.predict(X)

mae = mean_absolute_error(y, predicted_home_prices)

print("Mean Absolute Error using training data:", mae)

# STEP 6. Splitting Data into Training and Validation Sets
train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=0
)

# STEP 7. Building a Model Using Training Data
melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(train_X, train_y)

# STEP 8. Making Predictions on Validation Data
val_predictions = melbourne_model.predict(val_X)

# STEP 9. Evaluating Model Performance
val_mae = mean_absolute_error(val_y, val_predictions)

print("Mean Absolute Error using validation data:", val_mae)

