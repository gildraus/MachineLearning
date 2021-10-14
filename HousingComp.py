import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

housing_test_data_path = 'KaggleDatasets/test.csv'
housing_train_data_path = 'KaggleDatasets/train.csv'

test = pd.read_csv(housing_test_data_path)
train = pd.read_csv(housing_train_data_path)

print(test.describe())