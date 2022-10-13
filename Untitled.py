"""
Created on Wed Aug 10 00:30:44 2022

@author: osman
"""

import pandas as pd
import os, glob
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


#  loading and reading data
path = "C:/Users/osman/OneDrive/Desktop/Summer-2022/Data Warehs & Predictive Anlys/Assignments/"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
df = pd.read_csv(fullpath,sep=',')
print(df)

# get the first 5 records
print(df.head(5))

# get the data infor
df.info()

# get a descriptive statistics summary of a given dataframe
print(df.describe())

# get the columns names
df.columns

# get num of rows and columns 
df.shape

# check null values in df
df.isna().sum()

# check the missing data
sns.heatmap(df.isnull(),yticklabels=False, cbar=False,cmap='viridis')

# drop Bike_Model column, cuz it has the highest number of null values  
df.drop('Bike_Model', axis = 1, inplace = True)

# drop the columns that have missing values
#df.dropna(inplace=True)

# drop the following columns
removal_cols  = ['X', 'Y','OBJECTID','event_unique_id', 'Longitude', 'Latitude', 'ObjectId2']
df.drop(removal_cols, axis = 1, inplace = True)

# get the count of the report date of week
sns.countplot(x='Report_DayOfWeek', data=df)

# replace null values with other
df['Bike_Colour'].fillna('other', inplace=True)

# replace zero value
df['Cost_of_Bike'].replace(0, np.nan, inplace=True)  


# Check the average of all the numeric columns
pd.set_option('display.max_columns',100)
print(df.groupby('Status').mean())
df['Cost_of_Bike'].fillna(df['Cost_of_Bike'].mean(), inplace = True)


#Converting objects type columns into datetime
df['Occurrence_Date'] = pd.to_datetime(df['Occurrence_Date'])
df['Report_Date'] = pd.to_datetime(df['Report_Date'])

# get the unique Status
df['Status'].unique()

#there are there status, stolen, unknown and recovered. I merged unknown to recovered.
df['Status'].replace('STOLEN', 0, inplace=True)
df['Status'].replace(['UNKNOWN', 'RECOVERED'], 1, inplace=True)


# (STOLEN = 0) and (RECOVERED = 1)
df['Status'].value_counts()

# Although Status, is imbalanced


#


# convert cost to cost catagory
low = df['Cost_of_Bike'].quantile(.25)
average = df['Cost_of_Bike'].quantile(.5)
high = df['Cost_of_Bike'].quantile(.75)
df['cost_catag'] = np.where(df['Cost_of_Bike'] <= low, 'low', np.where((df['Cost_of_Bike'] > low) & (df['Cost_of_Bike'] <= average), 'average', np.where((df['Cost_of_Bike'] > average) & (df['Cost_of_Bike'] <= high), 'high', 'luxury')))


# Data visualization
df.groupby('Report_Year')['Status'].count().plot(kind='bar', figsize = (16,5))

# Most bicycles were steal
df.groupby( 'Occurrence_Month')['Status'].count().plot(kind='bar', figsize=(16,4))

# Top 10 Unsafe Neighbourhood
df[df['Status'] == 0].groupby('NeighbourhoodName')['Status'].count().reset_index(name='Count').sort_values('Count', ascending= False).set_index('NeighbourhoodName')[:10].plot(kind='bar', figsize=(16,4))

# Top 10 safe Neighbourhood
df[df['Status'] == 0].groupby('NeighbourhoodName')['Status'].count().reset_index(name='Count').sort_values('Count', ascending= True).set_index('NeighbourhoodName')[:10].plot(kind='bar', figsize=(16,4))

# Top 10 most stolen Bike Type
df[df['Status'] == 0].groupby('Bike_Type')['Status'].count().reset_index().set_index("Bike_Type").sort_values("Status", ascending = False)[:10].plot(kind='bar', figsize=(16,4))

# osman In july, the most cases were registed followed by August
df.groupby('NeighbourhoodName')['Status'].value_counts().reset_index(name='count').sort_values('count', ascending=False)

#drop NeighbourhoodName
df.drop('NeighbourhoodName', axis = 1, inplace = True)

# # Top Premises_Type, from where the most of the bikes steal

# In[73]:


df[df['Status'] == 0].groupby('Premises_Type')['Status'].count().reset_index().set_index('Premises_Type').plot(kind='bar', figsize=(16,4))


# In[74]:


#Outside is the area from where most the bicycles were stole


# In[75]:


categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
categorical_cols


# In[76]:


from sklearn.preprocessing import LabelEncoder


# In[77]:


df[['Primary_Offence',
 'Occurrence_Month',
 'Occurrence_DayOfWeek',
 'Report_Month',
 'Report_DayOfWeek',
 'Division',
 'City',
 'Hood_ID',
 'Location_Type',
 'Premises_Type',
 'Bike_Make',
 'Bike_Type',
 'Bike_Colour',
 'cost_catag']]


# In[78]:


import pickle
# city encoding
city_le = LabelEncoder()
df['City'] = city_le.fit_transform(df['City'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "City.pkl", 'wb') as city_file:
    pickle.dump(city_le, city_file)
    city_file.close()
    
#Occurrence_Month
Occurrence_Month_le = LabelEncoder()
df['Occurrence_Month'] = Occurrence_Month_le.fit_transform(df['Occurrence_Month'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "Occurrence_Month.pkl", 'wb') as Occurrence_Month_file:
    pickle.dump(Occurrence_Month_le, Occurrence_Month_file)
    Occurrence_Month_file.close()

 #Occurrence_DayOfWeek
Occurrence_DayOfWeek_le = LabelEncoder()
df['Occurrence_DayOfWeek'] = Occurrence_DayOfWeek_le.fit_transform(df['Occurrence_DayOfWeek'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "Occurrence_DayOfWeek.pkl", 'wb') as Occurrence_DayOfWeek_file:
    pickle.dump(Occurrence_DayOfWeek_le, Occurrence_DayOfWeek_file)
    Occurrence_DayOfWeek_file.close()

 #Report_Month
Report_Month_le = LabelEncoder()
df['Report_Month'] = Report_Month_le.fit_transform(df['Report_Month'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "Report_Month.pkl", 'wb') as Report_Month_file:
    pickle.dump(Report_Month_le, Report_Month_file)
    Report_Month_file.close()

 #Report_DayOfWeek

Report_DayOfWeek_le = LabelEncoder()
df['Report_DayOfWeek'] = Report_DayOfWeek_le.fit_transform(df['Report_Month'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "Report_DayOfWeek.pkl", 'wb') as Report_DayOfWeek_file:
    pickle.dump(Report_DayOfWeek_le, Report_DayOfWeek_file)
    Report_DayOfWeek_file.close()




#  Division
Division_le = LabelEncoder()
df['Division'] = Division_le.fit_transform(df['Division'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "Division.pkl", 'wb') as Division_file:
    pickle.dump(Division_le, Division_file)
    Division_file.close()

#City
City_le = LabelEncoder()
df['City'] = City_le.fit_transform(df['City'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "City.pkl", 'wb') as City_file:
    pickle.dump(City_le, City_file)
    City_file.close()


#  'Hood_ID
Hood_ID_le = LabelEncoder()
df['Hood_ID'] = Hood_ID_le.fit_transform(df['Hood_ID'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "Hood_ID.pkl", 'wb') as Hood_ID_file:
    pickle.dump(Hood_ID_le, Hood_ID_file)
    Hood_ID_file.close()
    
#  'Location_Type',
Location_Type_le = LabelEncoder()
df['Location_Type'] = Location_Type_le.fit_transform(df['Location_Type'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "Location_Type.pkl", 'wb') as Location_Type_file:
    pickle.dump(Location_Type_le, Location_Type_file)
    Location_Type_file.close()


#  'Premises_Type
Premises_Type_le = LabelEncoder()
df['Premises_Type'] = Premises_Type_le.fit_transform(df['Premises_Type'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "Premises_Type.pkl", 'wb') as Premises_Type_file:
    pickle.dump(Premises_Type_le, Premises_Type_file)
    Premises_Type_file.close()

#  'Bike_Make',
Bike_Make_le = LabelEncoder()
df['Bike_Make'] = Bike_Make_le.fit_transform(df['Bike_Make'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "Bike_Make.pkl", 'wb') as Bike_Make_file:
    pickle.dump(Bike_Make_le, Bike_Make_file)
    Bike_Make_file.close()
    
#  'Bike_Type',

Bike_Type_le = LabelEncoder()
df['Bike_Type'] = Bike_Type_le.fit_transform(df['Bike_Type'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "Bike_Type.pkl", 'wb') as Bike_Type_file:
    pickle.dump(Bike_Type_le, Bike_Type_file)
    Bike_Type_file.close()
    
#  'Bike_Colour
Bike_Colour_le = LabelEncoder()
df['Bike_Colour'] = Bike_Colour_le.fit_transform(df['Bike_Colour'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "Bike_Colour.pkl", 'wb') as Bike_Colour_file:
    pickle.dump(Bike_Colour_le, Bike_Colour_file)
    Bike_Colour_file.close()

#  'cost_catag'
cost_catag_le = LabelEncoder()
df['cost_catag'] = cost_catag_le.fit_transform(df['cost_catag'])

if not os.path.isdir("Model"):
    os.mkdir("Model")

with open("Model/" + "cost_catag.pkl", 'wb') as cost_catag_file:
    pickle.dump(cost_catag_le, cost_catag_file)
    cost_catag_file.close()


# In[79]:


df.drop(['Primary_Offence', 'cost_catag'], axis = 1, inplace = True)


# In[80]:


df.drop(['Occurrence_Date', 'Report_Date'], axis = 1, inplace = True)


# In[91]:


X, Y = df.drop('Status', axis=1), df['Status']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)


# In[92]:


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(x_train, y_train)

rf_y_pred = rf.predict(x_test)
print('Accuracy of RandomForest is:', accuracy_score(y_test, rf_y_pred))

# Random forest classifier with tuning
params_grid = {
    'n_estimators': range(10, 100, 10),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': range(1, 10),
    'min_samples_split': range(2, 10),
    'min_samples_leaf': range(1, 10),
}

rs = RandomizedSearchCV(
    rf,
    params_grid,
    n_iter=10,
    cv=10,
    scoring='accuracy',
    return_train_score=False,
    verbose=2,
    random_state=88)

search = rs.fit(x_train, y_train)

# best parameters & estimator
print("Best Params: ", search.best_params_)
print("Best estimators are: ", search.best_estimator_)

accuracy = search.best_score_ * 100
print(
    "Accuracy for training dataset with tuning is : {:.2f}%".format(accuracy))

# Training the Random Forest Classification model on the Training Set with best param
fine_tuned_model = search.best_estimator_.fit(x_train, y_train)

# save model

# Predicting the Test set results
rf_y_pred = fine_tuned_model.predict(x_test)
# predict_proba to return numpy array with two columns for a binary classification for N and P
rf_y_scores = fine_tuned_model.predict_proba(x_test)

print('Classification Report(N): \n',
      classification_report(y_test, rf_y_pred))
print('Confusion Matrix(N): \n', confusion_matrix(y_test, rf_y_pred))
print('Accuracy(N): \n', metrics.accuracy_score(y_test, rf_y_pred))

# Comparing the Real Values with Predicted Values
newdf = pd.DataFrame({'Real Values': y_test, 'Predicted Values': rf_y_pred})
print(newdf)


# In[140]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
dt = DecisionTreeClassifier(criterion='gini')
kf = KFold(10, shuffle=True, random_state=60)
score = cross_val_score(dt, x_train, y_train, cv=kf)


# In[142]:


dt = DecisionTreeClassifier(criterion='gini')


# In[148]:


dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)


# In[149]:


print('Classification Report(N): \n',
      classification_report(y_test, dt_pred))
print('Confusion Matrix(N): \n', confusion_matrix(y_test, dt_pred))
print('Accuracy(N): \n', metrics.accuracy_score(y_test, dt_pred))


# In[93]:



with open('model.pkl','wb') as f:
    pickle.dump(fine_tuned_model,f)


# In[151]:


# load
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)


# In[86]:


xx = ['Occurrence_Year', 'Occurrence_Month', 'Occurrence_DayOfWeek',
       'Occurrence_DayOfMonth', 'Occurrence_DayOfYear', 'Occurrence_Hour',
       'Report_Year', 'Report_Month', 'Report_DayOfWeek', 'Report_DayOfMonth',
       'Report_DayOfYear', 'Report_Hour', 'Division', 'Hood_ID',
       'Premises_Type', 'Bike_Make', 'Bike_Type', 'Bike_Speed', 'Bike_Colour',
       'Cost_of_Bike', 'City', 'Location_Type']


# In[88]:


for i in X.columns:
    if i not in xx:
        print(i)


# In[89]:


len(xx)


# In[90]:


len(X.columns)


# In[97]:


with open("Model/x_test.pkl", 'wb') as f:
    pickle.dump(x_test, f)


# In[98]:


with open("Model/y_test.pkl", 'wb') as f:
    pickle.dump(y_test, f)


# In[ ]:




