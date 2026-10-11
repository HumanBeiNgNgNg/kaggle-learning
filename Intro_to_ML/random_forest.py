import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Step 1: Load and prepare data
melbourne_file_path = 'Intro_to_ML/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis=0)

# Step 2: Select target and features
y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# Step 3: Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=0
)

# Step 4: Build and train the Random Forest model
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

# Step 5: Make predictions and evaluate the model
melb_preds = forest_model.predict(val_X)
mae = mean_absolute_error(val_y, melb_preds)

print("Random Forest Mean Absolute Error:", round(mae))