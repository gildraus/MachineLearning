import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

melbourne_data = pd.read_csv('melb_data.csv')
melbourne_data = melbourne_data.dropna(axis = 0)

print(melbourne_data.describe())
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

y = melbourne_data.Price
X = melbourne_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

model = DecisionTreeRegressor(random_state=0)

model.fit(train_X, train_y)

predictions = model.predict(val_X)
print(mean_absolute_error(val_y, predictions))