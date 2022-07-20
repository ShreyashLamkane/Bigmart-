#!/usr/bin/env python
# coding: utf-8

# In[199]:


#import the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[200]:


#Load the train and test dataset in pandas Dataframe
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


# In[201]:


#check no of rows and columns in train dataset
train.shape


# In[202]:


#Print the name of Columns in train dataset
train.columns


# In[203]:


#check the number of rows and columns in test dataset
test.shape


# In[204]:


#Print the name of columns in test dataset
test.columns


# In[205]:


#Combine test and train into onew file to perform Exploratory data Analysis
train["source"]="train"
test["source"]="test"
data=pd.concat([train, test], ignore_index=True,sort=False)
print(data.shape)


# In[206]:


data.head()


# In[207]:


#Describe function for numerical data summary
data.describe()


# In[208]:


#checking for missing value
data.isnull().sum()


# In[209]:


#The column Item_weight has 2439 missing values and the Outlet_size has around 4016.
#Item_Outlet_Sales has 5681 missing values , which we will predict using the model.


# In[210]:


#Print the unique values in the Item_Fat_Content column , where there are only two unique types of fat content in items:lowfat and regular
data["Item_Fat_Content"].unique()


# In[211]:


#Print the unique values in the Outlet_Establishment_Year column, where the data ranges from 1985 to 2009
data["Outlet_Establishment_Year"].unique()


# In[212]:


#Calculate the OutLet_Age
data["Outlet_Age"]=2018-data["Outlet_Establishment_Year"]
data.head(2)


# In[213]:


#Unique values in Outlet_Size
data["Outlet_Size"].unique()


# In[214]:


#Printing the count value of Item_Fat_Content column
data["Item_Fat_Content"].value_counts()


# In[215]:


#Print the count value of Outlet_Size
data["Outlet_Size"].value_counts()


# In[216]:


#Use the mode function to find out the most common value in Outlet_Size
print(data["Outlet_Size"].mode()[0])
#Two variables with missing values - Item_weight and Outlet_Size
#Replacing missing values in Outlet_Size with value "medium"
data["Outlet_Size"]=data["Outlet_Size"].fillna(data["Outlet_Size"].mode()[0])


# In[217]:


#Replace missing values in Item_Weight with the mean weight
data["Item_Weight"]=data["Item_Weight"].fillna(data["Item_Weight"].mean())


# In[218]:


#Plot a histogram to reveal the distribution of Item_Visibility column
data["Item_Visibility"].hist(bins=20)


# In[219]:


#Decting outliers
#An outliers is a data point that lies outside the overall pattern in distribution 
#A commonly used rule states  that a data point is an outlier if it is more than 1.5IQR above the third quartile or below the first quartile
#Using this , one can remove the outliers and output the resulting data in fill_data variable
#calculate the first quantile for Item_Visibility
Q1 =data["Item_Visibility"].quantile(0.25)


# In[220]:


#Calculate the second quantile 
Q3=data["Item_Visibility"].quantile(0.75)


# In[221]:


#Calculate the interquartile range (IQR)
IQR= Q3-Q1


# In[222]:


#Now that IQR range is know remove the outliers from the data
#The resulting data is stored in fill_data variable 
fill_data=data.query("(@Q1 - 1.5 * @IQR) <= Item_Visibility <= (@Q3 + 1.5 * @IQR)")


# In[223]:


#Display the data
fill_data.head(2)


# In[224]:


#Check the shape of the recruiting dataset without the outliers
fill_data.shape


# In[225]:


#Shape of the original dataset is 14204 rows and 14 columns with outliers
data.shape


# In[226]:


#Check the shape of resulting dataset without the outliers
fill_data.shape


# In[227]:


#shape of original dataset is
data.shape


# In[228]:


#Assign fill_data dataset to data Dataframe;
data=fill_data
data.shape


# In[229]:


#Modify Item_Visibility by converting the numerical values into the categories Low Visibility , Visibility, and High Visibility
data["Item_Visibility_bins"]=pd.cut(data["Item_Visibility"],[0.000, 0.065, 0.13, 0.2], labels=["Low Viz", "Viz", "High Viz"])


# In[230]:


#Print the count of Item_Visibility_bins
data["Item_Visibility_bins"].value_counts()


# In[231]:


#Replacing null value with low visibility
data["Item_Visibility_bins"]=data["Item_Visibility_bins"].replace(np.nan, "Low Viz", regex=True)


# In[232]:


#We Found types and differences in representation in categoriesof Item_Fat_Content variable
#This can be corrected using 
#Replace all other representation of low fat with low
data["Item_Fat_Content"]=data["Item_Fat_Content"].replace(["low fat", "LF"], "Low Fat")


# In[233]:


#Replace all all representation of reg with regular
data["Item_Fat_Content"] =data["Item_Fat_Content"].replace("reg", "Regular")


# In[234]:


#Print unique fat count values 
data["Item_Fat_Content"].unique()


# In[235]:


#Code all categorical variables as numeric  using "LabelEncoder" from sklearn's preprocessing module
#Initialize the label encoder
le=LabelEncoder()


# In[236]:


#Transform Item_Fat_Content
data["Item_Fat_Content"]=le.fit_transform(data["Item_Fat_Content"])


# In[237]:


#Transform Item_Visibility_bins
data["Item_Visibility_bins"]=le.fit_transform(data["Item_Visibility_bins"])


# In[238]:


#Transform Outlet_Size
data["Outlet_Size"]=le.fit_transform(data["Outlet_Size"])


# In[239]:


#Transform  Outlet_Location_Type
data["Outlet_Location_Type"]=le.fit_transform(data["Outlet_Location_Type"])


# In[240]:


#Print the unique values of Outlet_Type
data["Outlet_Type"].unique()


# In[241]:


#Create dummies for Outlet_Type
dummy=pd.get_dummies(data["Outlet_Type"])
dummy.head()


# In[242]:


#Explore the column Item_Identifier
data["Item_Identifier"]


# In[243]:


#As there are multiple values of food , nonconsumable items, and drinks with different numbers , combine the item type
data["Item_Identifier"].value_counts()


# In[244]:


#As multiple categories are present in Item_Identifier, reduce this by mapping 
data["Item_Type_Combined"]=data["Item_Identifier"].apply(lambda x: x[0:2])
data["Item_Type_Combined"]=data["Item_Type_Combined"].map({'FD': 'Food',
                                                          'NC': 'Non-Consumable',
                                                          'DR': 'Drink'})


# In[245]:


#Only three categories are present in an Item_Type_Combined column.
data["Item_Type_Combined"].value_counts()


# In[246]:


#Perform one-hot encoding  for all columns as the model works an numerical values and not an categorical values.
data=pd.get_dummies(data, columns=["Item_Fat_Content", "Outlet_Location_Type", "Outlet_Size", "Outlet_Type", "Item_Type_Combined"])


# In[247]:


data.shape


# In[248]:


data.dtypes


# In[ ]:





# In[249]:


import warnings
warnings.filterwarnings('ignore')
#Drop the columns which have been converted to different types

data.drop(["Item_Type", "Outlet_Establishment_Year"], axis=1, inplace=True)


#Divide the dataset created earlier into train and test datasets
train=data.loc[data["source"]=="train"]
test=data.loc[data["source"]=="test"]

#Drop unnecessary columns
test.drop(["Item_Outlet_Sales", "source"], axis=1, inplace=True)
train.drop(["source"], axis=1, inplace=True)

#Export modified versions of the files
train.to_csv("train_modified.csv", index=False)
test.to_csv("test_modified.csv", index=False)


# In[250]:


#Read the train _modified.csv and test_modified.csv dataset
train2=pd.read_csv("train_modified.csv")
test2=pd.read_csv("test_modified.csv")


# In[271]:


#print the data types of train2 column
train2.dtypes


# In[272]:


#Drop the irrelevant variables from train2 dataset
#Create the independent variable x_train and dependent variable y_train
X_train=train2.drop(["Item_Outlet_Sales","Outlet_Identifier", "Item_Identifier"], axis=1)
y_train=train2.Item_Outlet_Sales


# In[273]:


#Drop those irrelevant variable from test2 dataset
X_test = test2.drop({"Outlet_Identifier", "Item_Identifier"}, axis=1)


# In[274]:


X_test


# In[275]:


X_train.head(2)


# In[276]:


y_train.head(2)


# In[277]:


#Import sklearn Libraries for model selection
from sklearn import model_selection
from sklearn.linear_model import LinearRegression


# In[278]:


#Create a train and test split
xtrain, xtest, ytrain, ytest= model_selection.train_test_split(X_train, y_train, test_size=0.3, random_state=42)


# In[279]:


#Fit linear regression to the training dataset
lin=LinearRegression()


# In[280]:


lin.fit(xtrain, ytrain)


# In[281]:


#Find the coefficient and intercept of the line 
#Use strain and ytrain for linear regression
print(lin.coef_)
lin.intercept_


# In[282]:


#Predict the test set results of training data
predictions=lin.predict(xtest)
predictions


# In[283]:


import math


# In[284]:


#Find the RMSE for the model
print(math.sqrt(mean_squared_error(ytest, predictions)))


# In[285]:


#Predict the column Item_Outlet_Sales of test dataset
y_sales_pred = lin.predict(X_test)
y_sales_pred


# In[286]:


test_predictions=pd.DataFrame({
    'Item_Identifier': test2['Item_Identifier'],
    'Outlet_Identifier': test2['Outlet_Identifier'],
    'Item_Outlet_Sales': y_sales_pred
}, columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet', 'Item_Outlet_Sales']
)


# In[287]:


test_predictions


# In[ ]:




