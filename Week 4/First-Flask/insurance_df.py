## basically the model.py
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle

insurance_df = pd.read_csv("/Users/elissakuon/data_glacier_repos/First-Flask/insurance.csv")
insurance_df.head()

print(insurance_df.size)
print(insurance_df.shape)
print(insurance_df.keys())

# Checking for missing values and duplicated values 
insurance_df.isna().sum()
insurance_df.duplicated().sum()
insurance_df[insurance_df.duplicated(keep = False)]

# Checking on the skewness 
insurance_df.skew(axis = 0, skipna = True)

# Checking on the variable types 
insurance_df.info()
insurance_df.describe()
insurance_df["sex"].unique()
insurance_df["smoker"].unique()
insurance_df["region"].unique()

# Dummy Variable on Categorical Variables
insurance_df = pd.get_dummies(insurance_df, drop_first=True)
insurance_df.head()

# Building the Model 
## LINEAR REGRESSION
x = insurance_df.drop(['charges'], axis = 1)
y = insurance_df.charges
# Split between training and testing = 80/20
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2,random_state = 0)
lr = LinearRegression().fit(x_train,y_train)

# Get the R-squared value 
print(lr.score(x_test,y_test))

## POLYNOMIAL REGRESSION
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(x)
# Split between training and testing = 80/20
x_train,x_test,y_train,y_test = train_test_split(X_poly, y, test_size = 0.2,random_state = 0)
plr = LinearRegression().fit(x_train, y_train)

# Get the R-squared value 
print(plr.score(x_test, y_test))

## RANDOM FOREST REGRESSOR 
# Split between training and testing = 80/20
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2,random_state = 0)
rf_model = RandomForestRegressor(n_estimators = 1000, criterion = 'mse', random_state = 0)
rf = rf_model.fit(x_train, y_train)

rf_train_pred = rf_model.predict(x_train)
rf_test_pred = rf_model.predict(x_test)
print('R2 train data: %.3f'%(r2_score(y_train,rf_train_pred)))
print('R2 test data: %.3f'% (r2_score(y_test,rf_test_pred)))     

                      
# Creating a pickle file for Random Forest Regressor
filename = 'model.pkl'
pickle.dump(rf, open(filename, 'wb'))






