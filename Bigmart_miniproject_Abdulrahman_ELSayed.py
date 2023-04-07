#!/usr/bin/env python
# coding: utf-8

# # Note
# this code is written in a jupyter notebook enviroment, the csv files should be uploded before running the code.

# # Problem Statement
# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities,certain attributes of each product and store have been defined.
# 
# It is required to build a predictive model and find out the sales of each product at a particular store in order to help BigMart to understand the properties of products and stores which play a key role in increasing sales.
# 

# # Hypothesis Generation
# 
# It is important to sort out the data that will be valuable in our study before even looking at the dataset in order to be able to focus on the topic and avoid bias or mis-leading data.
# 
# The sole purpose of this study is to find what increase the sales of a certain product or decrease it.
# 
# Many factors could affect the product sales, such as: 
# * product properites: specs, its MPR, its price
# * Store properites : Location, Size , Stock 
# 

# # Loading Packages and Data

# In[84]:


import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter


#ploting libraries
import matplotlib.pyplot as plt 
import seaborn as sns

#feature engineering
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#train test split
from sklearn.model_selection import train_test_split

#metrics
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.model_selection  import cross_val_score as CVS


#ML models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso , Ridge
#installing xgboost
get_ipython().system(' pip install xgboost')

import xgboost as xgb


#default theme and settings
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
pd.options.display.max_columns

#warning hadle
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")



# In[85]:


#loading_data
train_data=pd.read_csv('Train.csv')
test_data=pd.read_csv('Test.csv')


# # Data Structure and Content
# 

# In[86]:


# explore the first 5 rows of train data
train_data.head()


# In[87]:


# explore the first 5 rows of test data
test_data.head()


# # Exploratory Data Analysis

# In[88]:


#exploring the train set

print (train_data.info())
print (train_data.describe())


# it is clear that both weight and Outlet_size columns has some null values which need to be treated

# In[89]:


#exploring the test set
print (test_data.info())
print (test_data.describe())


# # handling missing values

# In[90]:


#missing values in an descending order and their percentage
print("train:\n")
print(train_data.isnull().sum().sort_values(ascending=False),"\n\n",train_data.isnull().sum()/train_data.shape[0]*100)
print("\n\n test:\n")
print(test_data.isnull().sum().sort_values(ascending=False),"\n\n",test_data.isnull().sum()/test_data.shape[0]*100)


# missing values % in train and test sets are nearly the same
# 
# Since Outlet_Size is a categorical column, therefore we will impute missing values with Medium the mode value
#  

# In[91]:


print("train data mode, test data mode\n", [train_data['Outlet_Size'].mode().values[0]],"    ",[test_data['Outlet_Size'].mode().values[0]])


# In[92]:


#filling the missing values in outlet_values with the mode value

train_data['Outlet_Size']=train_data['Outlet_Size'].fillna(train_data['Outlet_Size'].mode().values[0])
test_data['Outlet_Size']=test_data['Outlet_Size'].fillna(test_data['Outlet_Size'].mode().values[0])
train_data['Outlet_Size'].isnull().sum(),test_data['Outlet_Size'].isnull().sum()


# Item_weight is a numerical column therefore we need to visualize it's distribution for a clearer display by using the boxplot in seaborn.
# 
# first we check the outliners

# In[93]:


sns.boxplot(data=train_data['Item_Weight'],orient="v")
plt.title("Weight Box Plot for train data")


# In[94]:


sns.boxplot(data=test_data['Item_Weight'],orient="v")
plt.title("Weight Box Plot for test data")


# from these graphes there are no outliers in the weight column so we can use the mean value to replace the missing values.

# In[95]:


#using mean values to replace missing values

train_data['Item_Weight']=train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean())
test_data['Item_Weight']=test_data['Item_Weight'].fillna(test_data['Item_Weight'].mean())
train_data['Item_Weight'].isnull().sum(),test_data['Item_Weight'].isnull().sum()


# In[96]:


#checking_duplicates
train_data.duplicated().sum(),test_data.duplicated().sum()


# In[97]:


train_data.info()


# In[98]:


#more exploration in categorical data columns

print(train_data['Item_Fat_Content'].value_counts(),"\n",train_data['Item_Type'].value_counts(),"\n",train_data['Outlet_Identifier'].value_counts(),"\n",train_data['Outlet_Size'].value_counts(),"\n",train_data['Outlet_Location_Type'].value_counts(),"\n",train_data['Outlet_Type'].value_counts())


# if we look at the "Item_Fat_Content" we will find that LF and Low fat will refer to the same type, also reg and Regular, we need to fix that.

# In[99]:


#replacing the value.
train_data['Item_Fat_Content'].replace(['LF','low fat','reg'],['Low Fat','Low Fat','Regular'],inplace=True)
test_data['Item_Fat_Content'].replace(['LF','low fat','reg'],['Low Fat','Low Fat','Regular'],inplace=True)
print(train_data['Item_Fat_Content'].value_counts(),"\n\n",test_data['Item_Fat_Content'].value_counts())


# # Data Visualization

# * Univirate Analysis

# First we study the categorical columns which are ('Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
#  'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type')

# In[100]:


sns.countplot(x='Item_Fat_Content', data=train_data)
plt.xlabel('Item_Fat_Content', fontsize=14)


# In[101]:


plt.figure(figsize=(30,10))
sns.countplot(x='Item_Type', data=train_data)
plt.xlabel('Item_Type', fontsize=14)


# In[102]:


plt.figure(figsize=(25,10))
sns.countplot(x='Outlet_Identifier', data=train_data)
plt.xlabel('Outlet_Identifier', fontsize=14)


# In[103]:


sns.countplot(x='Outlet_Size', data=train_data)
plt.xlabel('Outlet_Size', fontsize=14)


# In[104]:


sns.countplot(x='Outlet_Location_Type', data=train_data)
plt.xlabel('Outlet_Location_Type', fontsize=14)


# In[105]:


plt.figure(figsize=(20,10))
sns.countplot(x='Outlet_Type', data=train_data)
plt.xlabel('Outlet_Type', fontsize=14)


# From these graphs we notice the following:
# * in Item_Fat_Content column: the most items sold are low fat.
# * in Item_Type : Item types that are distictly popular are fruits and vegetables and snack foods and the ones which are least popular are breakfast and seafood.
# * in Outlet_Identifier : Sold items are evenly among outlets except OUT010 and OUT019 which are lower.
# * in Outlet_Size : Bigmart outlets are mostly medium sized in this data.
# * in Outlet_Location_Type:  The most common type is Tier3.
# * in Outlet_Type: the most popular outlet type is Supermarket Type1.
# 
# now we take a look at the numerical data.

# In[106]:


# comapring weights with the sales
sns.regplot(data=train_data,x='Item_Weight',y='Item_Outlet_Sales')


# In[107]:


# comapring item visibility with the sales
sns.regplot(data=train_data,x='Item_Visibility',y='Item_Outlet_Sales')


# In[108]:


# comapring item MPR with the sales
sns.regplot(data=train_data,x='Item_MRP',y='Item_Outlet_Sales')


# In[109]:


# seeing what year has the most oulets 
sns.countplot(x='Outlet_Establishment_Year', data=train_data)
plt.xlabel('Outlet_Establishment_Year', fontsize=14)


# From these figures we deduce that:
# * The slope of the line in the Weight vs Sales figure is neither negative nor positive so we can't decide weight's effect on the sales.
# * The slope of the line in Visibility vs Sales is negative so normaly we can say that as Visibility increase sales decrease, but if we took a closer look we find that around the (0,0) point the density of points with low sales are quite high so we need to take that in our consideration.
# * The slope of the line in MPR vs Sales is positive which leads to that the higher MPR the higher the sales.

# * Bivariate Analysis 

# In[110]:


#create a correlation matrix
print("train data correlation matrix","\n\n",train_data.corr(),"\n\n","test data correlation matrix","\n\n",test_data.corr())


# In[111]:


#visualizing the train data matrix with heat map
sns.heatmap(train_data.corr())


# In[112]:


#visualizing the test data matrix with heat map
sns.heatmap(test_data.corr())


# from these graphs we can make sure the Item_Outlet_Sales is highly correlated with Item_MRP.
# 
# we can also try to see the relation between some of the categorical columns with the Oultlet_Sales

# In[113]:


#Visualizing item types with sales
plt.figure(figsize=(30,10))
sns.barplot(x='Item_Type' ,y='Item_Outlet_Sales', data=train_data)


# In[114]:


#Visualizing outlet identifier with sales
plt.figure(figsize=(30,10))
sns.barplot(x='Outlet_Identifier' ,y='Item_Outlet_Sales', data=train_data)


# In[115]:


#Visualizing outlet type with sales
plt.figure(figsize=(20,10))
sns.barplot(x='Outlet_Type' ,y='Item_Outlet_Sales', data=train_data)


# In[116]:


#Visualizing outlet size with sales
plt.figure(figsize=(20,10))
sns.barplot(x='Outlet_Size' ,y='Item_Outlet_Sales', data=train_data)


# In[117]:


#Visualizing outlet location type with sales
plt.figure(figsize=(20,10))
sns.barplot(x='Outlet_Location_Type' ,y='Item_Outlet_Sales', data=train_data)


# From the Perivous graphs we can deduce that:
# 
# * There is small differences between the items sales.
# * Outlet identifiers OUT010 and OUT019 have the lowest sales and OUT027 has the highest sales.
# * Supermarket 3 has the best sales suprisingly as supermarket 1 has more products.
# * Medium and high store sizes have similar sales which are higher than the sales of small size.
# * Tier 2 and Tier 3 have similar sales which are higher than the sales of Tier 1.
# 

# In[118]:


train_data.head()


# In[119]:


train_data['Outlet_Type'].value_counts()


# # Feature Engineering
# 
# * If we look at the categorical columns, The columns Outlet_Establishment_Year, Item_Identifier and Outlet_Identifier don't have significant values so we will drop them.
# * all the ordinal variable columns should be labeled encoded
# * all of the nominal variable columns should be one hot encoded
# 
# ## Label Encoding of ordinal variable columns
# 
# we have :
# * Item_Fat_Content
# * Outlet_Size
# * Outlet_Location_Type

# In[120]:


#label encoding of ordinal variables
label_encod=['Item_Fat_Content','Outlet_Size','Outlet_Location_Type']
for i in label_encod:
    train_data[i]=LabelEncoder().fit_transform(train_data[i])
    test_data[i]=LabelEncoder().fit_transform(test_data[i])
    
train_data['Outlet_Location_Type'].value_counts()


# ## One hot encoding of nominal variable columns
# 
# we have:
# * Outlet_Type 
# * Item_Type

# In[121]:


#one hot encoding of nominal variables
one_hot_col=['Outlet_Type','Item_Type']

train_data_oh=pd.get_dummies(train_data[one_hot_col])
test_data_oh=pd.get_dummies(test_data[one_hot_col])


#adding the one hot encoded columns to the main columns

train_data_fea=pd.concat([train_data,train_data_oh],axis=1)
test_data_fea=pd.concat([test_data,test_data_oh],axis=1)

#dropping the un-needed columns

train_data_fea=train_data_fea.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Type','Item_Type'],axis=1)
test_data_fea=test_data_fea.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Type','Item_Type'],axis=1)

train_data_fea.head()


# In[122]:


test_data_fea.head()


# # Modeling
# ## Linear Regression
# 
# we need to predict the sales of the products and know what aspects will increase the sales of products.

# In[123]:


y=train_data_fea['Item_Outlet_Sales']
X=train_data_fea.drop('Item_Outlet_Sales',axis=1)
columns_names=X.columns


# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

# Create a Linear Regression model and fit it to the training data
lr_model=LinearRegression()
lr_model.fit(X_train,y_train)

# Make predictions on the testing data
y_pred=lr_model.predict(X_test)

# Evaluate the performance of the model
mse=MSE(y_test,y_pred)
rmse=np.sqrt(mse)
r2=R2(y_test,y_pred)

print("MSE=",mse,"\n","RMSE=",rmse,"\n","R2=",r2)

#viewin effect of each feature on the model
coef_lin=pd.Series(lr_model.coef_,columns_names).sort_values()

print(coef_lin)


# In[124]:


#Visualizing
sns.barplot(x=lr_model.coef_, y=columns_names)


# In[125]:


#saving results of the test model as a csv
Linear_Regression=pd.DataFrame({'y_test':y_test,'prediction':y_pred})
Linear_Regression.to_csv("Linear Regression.csv") 


# #  Regularized linear regression
# ## Lasso regularization 

# In[126]:


# Create a Lasso model and fit it to the training data
lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate the performance of the model
mse_lasso=MSE(y_test,y_pred_lasso)
rmse_lasso=np.sqrt(mse_lasso)
r2_lasso=R2(y_test,y_pred_lasso)

print("MSE_lasso=",mse_lasso,"\n","RMSE_lasso=",rmse_lasso,"\n","R2_lasso=",r2_lasso)

#viewin effect of each feature on the model
coef_lin_lasso=pd.Series(lasso_model.coef_,columns_names).sort_values()

print(coef_lin_lasso)


# In[127]:


#Visualizing
sns.barplot(x=lasso_model.coef_, y=columns_names)


# In[128]:


#saving results of the test model as a csv
Linear_Regression_lasso_reg=pd.DataFrame({'y_test':y_test,'prediction':y_pred_lasso})
Linear_Regression_lasso_reg.to_csv("Linear Regression_lasso.csv")


# ## Ridge regularization

# In[129]:


## Create a Ridge model and fit it to the training data

ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)

## Make predictions on the testing data

y_pred_ridge=ridge_model.predict(X_test)

## Evaluate the performance of the Ridge model

mse_ridge=MSE(y_test,y_pred_ridge)
rmse_ridge=np.sqrt(mse_ridge)
r2_ridge=R2(y_test,y_pred_ridge)

print("MSE_ridge=",mse_ridge,"\n","RMSE_ridge=",rmse_ridge,"\n","R2_ridge=",r2_ridge)

#viewin effect of each feature on the model
coef_lin_ridge=pd.Series(ridge_model.coef_,columns_names).sort_values()

print(coef_lin_ridge)


# In[130]:


#Visualizing
sns.barplot(x=ridge_model.coef_, y=columns_names)


# In[131]:


#saving results of the test model as a csv
Linear_Regression_ridge=pd.DataFrame({'y_test':y_test,'prediction':y_pred_ridge})
Linear_Regression_ridge.to_csv("Linear Regression_ridge.csv")


# ## RandomForest model

# In[132]:


# Create a RandomForest model and fit it to the training data
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_rf = rf_model.predict(X_test)

# Evaluate the performance of the RandomForest model
mse_rf=MSE(y_test,y_pred_rf)
rmse_rf=np.sqrt(mse_rf)
r2_rf=R2(y_test,y_pred_rf)

print("MSE_rf=",mse_rf,"\n","RMSE_rf=",rmse_rf,"\n","R2_rf=",r2_rf)

#viewin effect of each feature on the model
coef_rf=pd.Series(rf_model.feature_importances_,columns_names).sort_values()

print(coef_rf)


# In[133]:


#Visualizing
sns.barplot(x=rf_model.feature_importances_, y=columns_names)


# In[134]:


#saving results of the test model as a csv
Linear_Regression_randomforest=pd.DataFrame({'y_test':y_test,'prediction':y_pred_rf})
Linear_Regression_randomforest.to_csv("Linear Regression_randomforest.csv")


# ## XGBoost Model
# 

# In[135]:


# Create a DMatrix object for XGBoost
d_train=xgb.DMatrix(X_train,label=y_train) 
d_test=xgb.DMatrix(X_test)

# Set the hyperparameters for the XGBoost model
params = {
    "objective": "reg:squarederror",
    "eta": 0.14,
    "max_depth": 5,
    "min_child_weight": 7,
    "gamma": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread": 4,
    "seed": 42
}

# Train the XGBoost model
xgb_model=xgb.train(params, d_train, num_boost_round=100)

# Make predictions on the testing data
y_pred_xgb = xgb_model.predict(d_test)


# Evaluate the performance of the model
mse_xgb=MSE(y_test,y_pred_xgb)
rmse_xgb=np.sqrt(mse_xgb)
r2_xgb=R2(y_test,y_pred_xgb)

print("MSE_xgb=",mse_xgb,"\n","RMSE_xgb=",rmse_xgb,"\n","R2_xgb=",r2_xgb)



# In[136]:


#Visualizing the importance of each Feature
plt.figure(figsize=(20,30))
xgb.plot_importance(xgb_model)
plt.show()


# In[137]:


#saving results of the test model as a csv
Linear_Regression_xgboost=pd.DataFrame({'y_test':y_test,'prediction':y_pred_xgb})
Linear_Regression_xgboost.to_csv("Linear Regression_xgboost.csv")


# In[138]:


#summarizing the performance of each model
MAE= [mse,mse_lasso,mse_ridge,mse_rf,mse_xgb]
MSE= [rmse,rmse_lasso,rmse_ridge,rmse_rf,rmse_xgb]
R_2= [r2,r2_lasso,r2_ridge,r2_rf,r2_xgb]

Models = pd.DataFrame({
    'models': ["Linear Regression","Lasso Regressor","Ridge Regressor","Random Forest Regressor","XGBoost model"],
    'MAE': MAE, 'MSE': MSE, 'R^2':R_2,})
Models.sort_values(by='R^2', ascending=True)


# # Conclusion
# 
# 
# * Item_MRP optimizes Maximum Outlet sales (positive correlation with the target).
# * XGBoost model and Lasso Regressor have the best perfomance in most categories.
# 
