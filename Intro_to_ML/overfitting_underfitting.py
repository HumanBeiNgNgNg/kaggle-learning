import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
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

# Step 4: Create a function to calculate MAE
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        random_state=0
    )
    model.fit(train_X, train_y)

    val_predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, val_predictions)

    return mae

# Step 5: Compare models with different numbers of leaf nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print(
        f"Max leaf nodes: {max_leaf_nodes}\t"
        f"Mean Absolute Error: {mae:.0f}"
    )
