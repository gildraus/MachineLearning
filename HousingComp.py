import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sb

def sum_nullova_kolone():
    brojac = 0
    for x in col:
        flag = train[x].isnull().values.any()
        if flag == True:
            print(x)
            brojac = brojac + 1
    print('Ima ' + str(brojac) + ' kolona sa nan vrednostima ')

housing_train_data_path = 'KaggleDatasets/train.csv'
housing_test_data_path = 'KaggleDatasets/test.csv'

train = pd.read_csv(housing_train_data_path)
test = pd.read_csv(housing_test_data_path)

col = train.columns

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

#Electrical
train = train.dropna(subset = ['Electrical'])

#FireplaceQu
train['FireplaceQu'] = train['FireplaceQu'].fillna('NF')

#GarageType
train['GarageType'] = train['GarageType'].fillna('NG')

#GarageYrBlt
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].median())

#GarageFinish
train['GarageFinish'] = train['GarageFinish'].fillna('NG')

#GarageQual
train['GarageQual'] = train['GarageQual'].fillna('NG')

#GarageCond
train['GarageCond'] = train['GarageCond'].fillna('NG')

#PoolQC
train['PoolQC'] = train['PoolQC'].fillna('NP')\

#Fence
train['Fence'] = train['Fence'].fillna('NF')

#MiscFeature - irrelevant - should remove it but don't know how xD
train['MiscFeature'] = train['MiscFeature'].fillna('None')


sum_nullova_kolone()
















