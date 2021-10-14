import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sb

housing_test_data_path = 'KaggleDatasets/test.csv'
housing_train_data_path = 'KaggleDatasets/train.csv'

test = pd.read_csv(housing_test_data_path)
train = pd.read_csv(housing_train_data_path)


#stampamo dimenzije matrice
print(train.shape)
#prebrojavamo null vrednosti
print(train.isnull().sum())

sb.heatmap(train.isnull(),yticklabels=False, cbar = False)
print('s')