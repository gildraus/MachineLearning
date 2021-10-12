import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
#print(melbourne_data.describe())

#print(melbourne_data.columns)

melbourne_data =melbourne_data.dropna(axis = 0)
#print(melbourne_data.describe())
#print(melbourne_data.columns)

#print(melbourne_data.Price)

#prediction target(y) = the column we want to predict
y = melbourne_data.Price

#features(X) =  the columns used to determine the home price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

#dataframe.describe() same as dataframe.summary() in R
#print(X.describe())
#printing 5 rows
#print(X.head())

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

#print("Making predictions for the following 5 houses:")
#print(X.head())
#print("The predictions are")
#print(melbourne_model.predict(X.head()))

###
#
#
# MODEL VALIDATION
#
#
# ###

from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mae = mean_absolute_error(y, predicted_home_prices)
#but result is IN-SAMPLE mean absolute error which is bad
#print(mae)

from sklearn.model_selection import train_test_split

###
# PODSETNIK:
# y - izlazna varijabla(u ovom slucaju SalePrice)
# X - dataframe koji ima sve(ili sve bitne za model) varijable osim izlazne
# ###

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#random_state(Python) = set.seed(R)


