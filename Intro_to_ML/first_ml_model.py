import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# STEP 1. Selecting Data for Modeling
melbourne_file_path = 'Intro_to_ML/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
print(melbourne_data.columns)

'''
The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
We'll learn to handle missing values in a later tutorial.  
Your Iowa data doesn't have missing values in the columns you use. 
So we will take the simplest option for now, and drop houses from our data. 
Don't worry about this much for now, though the code is:
'''
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# STEP 2. Selecting The Prediction Target
y = melbourne_data.Price

# STEP 3. Choosing "Features"
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
X.describe()
X.head()

# STEP 4. Building Your Model
'''
The steps to building and using a model are:
    Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
    Fit: Capture patterns from provided data. This is the heart of modeling.
    Predict: Just what it sounds like
    Evaluate: Determine how accurate the model's predictions are
'''
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

