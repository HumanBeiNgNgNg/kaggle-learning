import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Step 1: Load and prepare data
melbourne_file_path = 'Intro_to_ML/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis=0)

# Step 2: Select target and features
y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# Step 3: Build and train model
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

# Step 4: Make predictions
print("Predictions for the first 5 houses:")
print(melbourne_model.predict(X.head()))

