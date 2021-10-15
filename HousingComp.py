import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sb

def sum_nullova_kolone_train():
    brojac = 0
    for x in col_train:
        flag = train[x].isnull().values.any()
        if flag == True:
            print(x)
            brojac = brojac + 1
    print('Ima ' + str(brojac) + ' train kolona sa nan vrednostima ')

def sum_nullova_kolone_test():
    brojac = 0
    for x in col_test:
        flag = test[x].isnull().values.any()
        if flag == True:
            print(x)
            brojac = brojac + 1
    print('Ima ' + str(brojac) + ' test kolona sa nan vrednostima ')

housing_train_data_path = 'KaggleDatasets/train.csv'
housing_test_data_path = 'KaggleDatasets/test.csv'

train = pd.read_csv(housing_train_data_path)
test = pd.read_csv(housing_test_data_path)

col_train = train.columns
col_test = test.columns

#sredjujemo LotFrontage svaki koji ima null
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())

#sredjujemo Alley
#posto pise da je NA zapravo no alley access ja sam stavio da za svaki NaN
#vazi da to znaci da zapravo nema alley acces odnosno Naac
train['Alley'] = train['Alley'].fillna('Naac')
test['Alley'] = test['Alley'].fillna('Naac')

#Masonry veneer type
train['MasVnrType'] = train['MasVnrType'].fillna('None')
test['MasVnrType'] = test['MasVnrType'].fillna('None')

#Masonry veneer area in square feet
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())


#BsmtQual
train['BsmtQual'] = train['BsmtQual'].fillna('NoB')
test['BsmtQual'] = test['BsmtQual'].fillna('NoB')

#BsmtCond
train['BsmtCond'] = train['BsmtCond'].fillna('NoB')
test['BsmtCond'] = test['BsmtCond'].fillna('NoB')

#BsmtExposure
train['BsmtExposure'] = train['BsmtExposure'].fillna('NoB')
test['BsmtExposure'] = test['BsmtExposure'].fillna('NoB')

#BsmtFinType1
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('NoB')
test['BsmtFinType1'] = test['BsmtFinType1'].fillna('NoB')

#BsmtFinType2
train['BsmtFinType2'] = train['BsmtFinType2'].fillna('NoB')
test['BsmtFinType2'] = test['BsmtFinType2'].fillna('NoB')

#Electrical
train = train.dropna(subset = ['Electrical'])
test = test.dropna(subset = ['Electrical'])

#FireplaceQu
train['FireplaceQu'] = train['FireplaceQu'].fillna('NF')
test['FireplaceQu'] = test['FireplaceQu'].fillna('NF')

#GarageType
train['GarageType'] = train['GarageType'].fillna('NG')
test['GarageType'] = test['GarageType'].fillna('NG')

#GarageYrBlt
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].median())
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].median())

#GarageFinish
train['GarageFinish'] = train['GarageFinish'].fillna('NG')
test['GarageFinish'] = test['GarageFinish'].fillna('NG')

#GarageQual
train['GarageQual'] = train['GarageQual'].fillna('NG')
test['GarageQual'] = test['GarageQual'].fillna('NG')

#GarageCond
train['GarageCond'] = train['GarageCond'].fillna('NG')
test['GarageCond'] = test['GarageCond'].fillna('NG')

#PoolQC
train['PoolQC'] = train['PoolQC'].fillna('NP')
test['PoolQC'] = test['PoolQC'].fillna('NP')

#Fence
train['Fence'] = train['Fence'].fillna('NF')
test['Fence'] = test['Fence'].fillna('NF')

#MiscFeature - irrelevant - should remove it but don't know how xD
train['MiscFeature'] = train['MiscFeature'].fillna('None')
test['MiscFeature'] = test['MiscFeature'].fillna('None')


#MSZoning
test = test.dropna(subset = ['MSZoning'])

#Utilities
test = test.dropna(subset = ['Utilities'])

#Exterior1st
test = test.dropna(subset = ['Exterior1st'])

#Exterior2nd
test = test.dropna(subset = ['Exterior2nd'])

#BsmtFinSF1
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())

#BsmtFinSF2
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())

#BsmtUnfSF
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())

#TotalBsmtSF
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())

#BsmtFullBath
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].median())

#BsmtHalfBath
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].median())

#KitchenQual
test = test.dropna(subset = ['KitchenQual'])

#Functional
test = test.dropna(subset = ['Functional'])

#GarageCars
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].median())

#GarageArea
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())

#SaleType
test = test.dropna(subset = ['SaleType'])

sum_nullova_kolone_train()
sum_nullova_kolone_test()
















