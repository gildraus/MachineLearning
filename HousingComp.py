import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sb

housing_test_data_path = 'KaggleDatasets/test.csv'
housing_train_data_path = 'KaggleDatasets/train.csv'

test = pd.read_csv(housing_test_data_path)
train = pd.read_csv(housing_train_data_path)




col = train.columns
print(col)




#sredjujemo LotFrontage svaki koji ima null
mean_lot_frontage = train['LotFrontage'].mean()
train['LotFrontage'] = train['LotFrontage'].fillna(mean_lot_frontage)

#sredjujemo Alley
#posto pise da je NA zapravo no alley access ja sam stavio da za svaki NaN
#vazi da to znaci da zapravo nema alley acces odnosno Naac
train['Alley'] = train['Alley'].fillna('Naac')

#Masonry veneer type
train['MasVnrType'] = train['MasVnrType'].fillna('None')

#Masonry veneer area in square feet
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())

#BsmtQual
train['BsmtQual'] = train['BsmtQual'].fillna('NoB')

#BsmtCond
train['BsmtCond'] = train['BsmtCond'].fillna('NoB')

#BsmtExposure
train['BsmtExposure'] = train['BsmtExposure'].fillna('NoB')

#BsmtFinType1
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('NoB')

#BsmtFinType2
train['BsmtFinType2'] = train['BsmtFinType2'].fillna('NoB')



#prebrojavamo null vrednosti
#print(train.isnull().sum())
###
#
brojac = 0
for x in col:
    flag = train[x].isnull().values.any()
    if flag == True:
        print(x)
        brojac = brojac + 1
print('Ima ' + str(brojac) + ' kolona sa nan vrednostima ')















