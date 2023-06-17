#!/usr/bin/env python
# coding: utf-8

# # Used Car Price Prediction
# ## Life cycle of Machine learning Project
# 
# - Understanding the Problem Statement
# - Data Collection
# - Exploratory data analysis
# - Data Cleaning
# - Data Pre-Processing
# - Model Training
# - Choose best model
# ## 1) Problem statement.
# - This dataset comprises used cars sold on cardehko.com in India as well as important features of these cars.
# - If user can predict the price of the car based on input features.
# Prediction results can be used to give new seller the price suggestion based on market condition.
# ## 2) Data Collection.
# - The Dataset is collected from scrapping from cardheko webiste
# - The data consists of 13 column and 15411 rows.
# ### 2.1 Import Data and Required Packages
# ### Importing Pandas, Numpy, Matplotlib, Seaborn and Warings Library.

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
from six.moves import urllib
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Download and Import the CSV Data as Pandas DataFrame

# In[71]:


df = pd.read_csv("D:/I neuron/car_dekho.csv")


# In[72]:


df.head()


# In[8]:


df.shape


# ### summary of the given dataset

# In[9]:


df.describe()


# ### checking of data types

# In[10]:


df.info()


# ## 3. EXPLORING DATA

# In[17]:


# define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))


# ## Feature Information
# car_name: Car's Full name, which includes brand and specific model name.  
# brand: Brand Name of the particular car.  
# model: Exact model name of the car of a particular brand.  
# seller_type: Which Type of seller is selling the used car  
# fuel_type: Fuel used in the used car, which was put up on sale.  
# transmission_type: Transmission used in the used car, which was put on sale.  
# vehicle_age: The count of years since car was bought.  
# mileage: It is the number of kilometer the car runs per litre.  
# engine: It is the engine capacity in cc(cubic centimeters)  
# max_power: Max power it produces in BHP.  
# seats: Total number of seats in car.  
# selling_price: The sale price which was put up on website.  

# In[18]:


# proportion of count data on categorical columns
for col in categorical_features:
    print(df[col].value_counts(normalize=True) * 100)
    print('---------------------------')


# ### Univariate Analysis

# The term univariate analysis refers to the analysis of one variable prefix “uni” means “one.” The purpose of univariate analysis is to understand the distribution of values for a single variable.

# ## Numerical features

# In[19]:


plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(numeric_features)):
    plt.subplot(5, 3, i+1)
    sns.kdeplot(x=df[numeric_features[i]],shade=True, color='b')
    plt.xlabel(numeric_features[i])
    plt.tight_layout()


# ### Points Taken
# - Km_driven, max_power, selling_price, and engine are right skewed and postively skewed.  
# - Outliers are found in Km_driven, max_power, selling_price, and engine.  

# ### Categorical Features

# In[21]:


# categorical columns
plt.figure(figsize=(20, 15))
plt.suptitle('Univariate Analysis of Categorical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
cat1 = [ 'brand', 'seller_type', 'fuel_type', 'transmission_type']
for i in range(0, len(cat1)):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=df[cat1[i]])
    plt.xlabel(cat1[i])
    plt.xticks(rotation=45)
    plt.tight_layout()


# ### Points taken
# - Maruti and hyundai are highest sellers
# - selling at dealers end is highest
# - Manual transmission vehicle is more

# ### Multivariate Analysis
# - Multivariate analysis is the analysis of more than one variable.

# ### Check Multicollinearity in Numerical features

# In[24]:


df[(list(df.columns)[1:])].corr()


# In[25]:


plt.figure(figsize = (15,10))
sns.heatmap(df.corr(), cmap="CMRmap", annot=True)
plt.show()


# ### Points taken
# 
# - Our target column mileage has a weak negative correlation with engine.  
# - The max_power has a strong positive correlation with engine.  
# 

# ### Check Multicollinearity for Categorical features
# - A chi-squared test (also chi-square or χ2 test) is a statistical hypothesis test that is valid to perform when the test statistic is chi-squared distributed under the null hypothesis, specifically Pearson's chi-squared test
# 
# - A chi-square statistic is one way to show a relationship between two categorical variables.
# 
# - Here we test correlation of Categorical columns with Target column i.e Selling Price

# In[26]:


from scipy.stats import chi2_contingency
chi2_test = []
for feature in categorical_features:
    if chi2_contingency(pd.crosstab(df['selling_price'], df[feature]))[1] < 0.05:
        chi2_test.append('Reject Null Hypothesis')
    else:
        chi2_test.append('Fail to Reject Null Hypothesis')
result = pd.DataFrame(data=[categorical_features, chi2_test]).T
result.columns = ['Column', 'Hypothesis Result']
result


# ### Checking Null Values

# In[27]:


df.isnull().sum()


# In[28]:


continues_features=[feature for feature in numeric_features if len(df[feature].unique())>=10]
print('Num of continues features :',continues_features)


# In[30]:


fig = plt.figure(figsize=(15, 20))

for i in range(0, len(continues_features)):
    ax = plt.subplot(8, 2, i+1)

    sns.scatterplot(data= df ,x='selling_price', y=continues_features[i], color='b')
    plt.xlim(0,25000000) # Limit to 20 lakhs Rupees to view clean
    plt.tight_layout()


# ### Initial Analysis Report
# #### Report
# 
# - Lower Vehicle age has more selling price than Vehicle with more age.  
# - Engine CC has positive effect on price,Vehicle with 2000 cc and below are mostly priced below 5lakh.  
# - Kms Driven has negative effect on selling price.  

# ## 4. Visualization
# ### 4.1 Visualize the Target Feature

# In[31]:


plt.subplots(figsize=(14,7))
sns.histplot(df.selling_price, bins=200, kde=True, color = 'b')
plt.title("Selling Price Distribution", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=12)
plt.xlabel("Selling price in millions", weight="bold", fontsize=12)
plt.xlim(0,3000000)
plt.show()


# - From the chart it is clear that the Target Variable Skewed  
# ### 4.2 Most Selling car in Used car website?

# In[33]:


df.car_name.value_counts()[0:10]


# ### Point Taken
# - Most selling car is Hyundai i20

# In[35]:


plt.subplots(figsize=(14,7))
sns.countplot(x="car_name", data=df,ec = "black",palette="Set1",order = df['car_name'].value_counts().index)
plt.title("Top 10 Most Sold Car", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=20)
plt.xlabel("Car Name", weight="bold", fontsize=16)
plt.xticks(rotation= 45)
plt.xlim(-1,10.5)
plt.show()


# ### Check mean price of Hyundai i20 which is most sold

# In[36]:


i20 = df[df['car_name'] == 'Hyundai i20']['selling_price'].mean()
print(f'The mean price of Hyundai i20 is {i20:.2f} Rupees')


# ### Point Taken:
# 
# - As per the Chart these are top 10 most selling cars in used car website.  
# - Of the total cars sold Hyundai i20 shares 5.8% of total ads posted and followed by Maruti Swift Dzire.  
# - Mean Price of Most Sold Car is 5.4 lakhs.  
# - This Feature has impact on the Target Variable.  

# ## Most selling brand

# In[38]:


df.brand.value_counts()[0:10]


# In[39]:


plt.subplots(figsize=(14,7))
sns.countplot(x="brand", data=df,ec = "black",palette="Set2",order = df['brand'].value_counts().index)
plt.title("Top 10 Most Sold Brand", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=14)
plt.xlabel("Brand", weight="bold", fontsize=16)
plt.xticks(rotation= 45)
plt.xlim(-1,10.5)
plt.show()


# ## Check the Mean price of Maruti brand which is most sold

# In[41]:


maruti = df[df['brand'] == 'Maruti']['selling_price'].mean()
print(f'The mean price of Maruti is {maruti:.3f} Rupees')


# ### Report:
# 
# - As per the Chart Maruti has the most share of Ads in Used car website and Maruti is the most sold brand.
# - Following Maruti we have Hyundai and Honda.
# - Mean Price of Maruti Brand is 4.8 lakhs.

# ### Costliest Brand and Costliest Car

# In[44]:


brand = df.groupby('brand').selling_price.max()
brand_df = brand.to_frame().sort_values('selling_price',ascending=False)[0:10]
brand_df


# In[45]:


plt.subplots(figsize=(14,7))
sns.barplot(x=brand.index, y=brand.values,ec = "black",palette="Set2")
plt.title("Brand vs Selling Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price", weight="bold", fontsize=15)
plt.xlabel("Brand Name", weight="bold", fontsize=16)
plt.xticks(rotation=90)
plt.show()


# ### Point Taken:
# 
# - Costliest Brand sold is Ferrari at 3.95 Crores.
# - Second most costliest car Brand is Rolls-Royce as 2.42 Crores.
# - Brand name has very clear impact on selling price.

# ### Costliest Car

# In[46]:


car= df.groupby('car_name').selling_price.max()
car =car.to_frame().sort_values('selling_price',ascending=False)[0:10]
car


# In[47]:


plt.subplots(figsize=(14,7))
sns.barplot(x=car.index, y=car.selling_price,ec = "black",palette="Set1")
plt.title("Car Name vs Selling Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price", weight="bold", fontsize=15)
plt.xlabel("Car Name", weight="bold", fontsize=16)
plt.xticks(rotation=90)
plt.show()


# ### Point Taken
# 
# - Costliest Car sold is Ferrari GTC4 Lusso followed by Rolls Royce Ghost.
# - Ferrari selling price is 3.95 Crs.
# - Other than Ferrari other car has priced below 1.5cr.

# ### Most Mileage Brand and Car Name

# In[49]:


mileage= df.groupby('brand')['mileage'].mean().sort_values(ascending=False).head(15)
mileage.to_frame()


# In[50]:


plt.subplots(figsize=(14,7))
sns.barplot(x=mileage.index, y=mileage.values, ec = "black", palette="Set2")
plt.title("Brand vs Mileage", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mileage in Kmpl", weight="bold", fontsize=15)
plt.xlabel("Brand Name", weight="bold", fontsize=12)
plt.ylim(0,25)
plt.xticks(rotation=45)
plt.show()


# ### Car with Highest Mileage

# In[51]:


mileage_C= df.groupby('car_name')['mileage'].mean().sort_values(ascending=False).head(10)
mileage_C.to_frame()


# In[52]:


plt.subplots(figsize=(14,7))
sns.barplot(x=mileage_C.index, y=mileage_C.values, ec = "black", palette="Set1")
plt.title("Car Name vs Mileage", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mileage in Kmpl", weight="bold", fontsize=15)
plt.xlabel("Car Name", weight="bold", fontsize=12)
plt.ylim(0,27)
plt.xticks(rotation=45)
plt.show()


# ## Kilometer driven vs Selling Price

# In[53]:


plt.subplots(figsize=(14,7))
sns.scatterplot(x="km_driven", y='selling_price', data=df,ec = "white",color='b', hue='fuel_type')
plt.title("Kilometer Driven vs Selling Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price", weight="bold", fontsize=20)
plt.xlim(-10000,800000) #used limit for better visualization
plt.ylim(-10000,10000000)
plt.xlabel("Kilometer driven", weight="bold", fontsize=16)
plt.show()


# ### Points taken
# 
# - Many Cars were sold with kms between 0 to 20k Kilometers
# - Low Kms driven cars had more selling price compared to cars which had more kms driven.
# 
# ## Fuel Type Selling Price

# In[54]:


fuel = df.groupby('fuel_type')['selling_price'].median().sort_values(ascending=False)
fuel.to_frame()


# In[55]:


plt.subplots(figsize=(14,7))
sns.barplot(x=df.fuel_type, y=df.selling_price, ec = "black", palette="Set2_r")
plt.title("Fuel type vs Selling Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price Median", weight="bold", fontsize=15)
plt.xlabel("Fuel Type", weight="bold", fontsize=12)
plt.show()


# ### Points taken
# 
# - Electric cars have highers selling average price.
# - Followed by Diesel and Petrol.
# - Fuel Type is also important feature for the Target variable

# ### Most sold Fuel type

# In[56]:


plt.subplots(figsize=(14,7))
sns.countplot(x=df.fuel_type, ec = "black", palette="Set2_r")
plt.title("Fuel Type Count", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=15)
plt.xlabel("Fuel Type", weight="bold", fontsize=12)
plt.show()


# ### Points taken
# 
# - Petrol and Diesel dominate the used car market in the website.
# - The most sold fuel type Vechicle is Petrol.
# - Followed by diesel and CNG and least sold is Electric
# 
# ## Fuel types available and mileage given

# In[57]:


fuel_mileage = df.groupby('fuel_type')['mileage'].mean().sort_values(ascending=False)
fuel_mileage.to_frame()


# In[58]:


plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type', y='mileage', data=df,palette="Set1_r")
plt.title("Fuel type vs Mileage", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mileage in Kmpl", weight="bold", fontsize=15)
plt.xlabel("Fuel Type", weight="bold", fontsize=12)
plt.show()


# ## Mileage vs Selling Price

# In[59]:


plt.subplots(figsize=(14,7))
sns.scatterplot(x="mileage", y='selling_price', data=df,ec = "white",color='b', hue='fuel_type')
plt.title("Mileage vs Selling Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price", weight="bold", fontsize=20)
plt.ylim(-10000,10000000)
plt.xlabel("Mileage", weight="bold", fontsize=16)
plt.show()


# In[60]:


plt.subplots(figsize=(14,7))
sns.histplot(x=df.mileage, ec = "black", color='g', kde=True)
plt.title("Mileage Distribution", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=15)
plt.xlabel("Mileage", weight="bold", fontsize=12)
plt.show()


# ## Vehicle age vs Selling Price

# In[61]:


plt.subplots(figsize=(20,10))
sns.lineplot(x='vehicle_age',y='selling_price',data=df,color='b')
plt.ylim(0,2500000)
plt.show()


# ### Points taken
# 
# - As the Vehicle age increases the price also get reduced.
# - Vehicle age has Negative impact on selling price
# 
# ## Vehicle age vs Mileage

# In[62]:


vehicle_age = df.groupby('vehicle_age')['mileage'].median().sort_values(ascending=False)
vehicle_age.to_frame().head(5)


# In[63]:


plt.subplots(figsize=(14,7))
sns.boxplot(x=df.vehicle_age, y= df.mileage, palette="Set1")
plt.title("Vehicle Age vs Mileage", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mileage", weight="bold", fontsize=20)
plt.xlabel("Vehicle Age in Years", weight="bold", fontsize=16)
plt.show()


# ### Point taken
# 
# - As the Age of vehicle increases the median of mileage drops.
# - Newer Vehicles have more mileage median older vehicle.

# In[65]:


oldest = df.groupby('car_name')['vehicle_age'].max().sort_values(ascending=False).head(10)
oldest.to_frame()


# ### Point taken
# 
# - Maruti Alto is the Oldest car available 29 years old in the used car website followed by BMW 3 for 25 years old.
# 
# ## Transmission Type

# In[66]:


plt.subplots(figsize=(14,7))
sns.countplot(x='transmission_type', data=df,palette="Set1")
plt.title("Transmission type Count", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=15)
plt.xlabel("Transmission Type", weight="bold", fontsize=12)
plt.show() 


# In[67]:


plt.subplots(figsize=(14,7))
sns.barplot(x='transmission_type', y='selling_price', data=df,palette="Set1")
plt.title("Transmission type vs Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price in Millions", weight="bold", fontsize=15)
plt.xlabel("Transmission Type", weight="bold", fontsize=12)
plt.show() 


# ### Point taken
# 
# - Manual Transmission was found in most of the cars which was sold.
# - Automatic cars have more selling price than manual cars.
# 
# ## Seller Type

# In[68]:


plt.subplots(figsize=(14,7))
sns.countplot(x='seller_type', data=df,palette="rocket_r")
plt.title("Transmission type vs Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price in Millions", weight="bold", fontsize=15)
plt.xlabel("Transmission Type", weight="bold", fontsize=12)
plt.show() 


# In[69]:


dealer = df.groupby('seller_type')['selling_price'].median().sort_values(ascending=False)
dealer.to_frame()


# ### Points Taken
# 
# - Dealers have put more ads on used car website.
# - Dealers have put 9539 ads with median selling price of 5.91 Lakhs.
# - Followed by Individual with 5699 ads with median selling price of 5.4 Lakhs.
# - Dealers have more median selling price than Individual.
# 
# ### Final Findings
# 
# - The datatypes and Column names were right and there was 15411 rows and 13 columns
# - The selling_price column is the target to predict. i.e Regression Problem.
# - There are outliers in the km_driven, enginer, selling_price, and max power.
# - Dealers are the highest sellers of the used cars.
# - Skewness is found in few of the columns will check it after handling outliers.
# - Vehicle age has negative impact on the price.
# - Manual cars are mostly sold and automatic has higher selling average than manual cars.
# - Petrol is the most preffered choice of fuel in used car website, followed by diesel and LPG.
# - We just need less data cleaning for this dataset.

# In[ ]:




