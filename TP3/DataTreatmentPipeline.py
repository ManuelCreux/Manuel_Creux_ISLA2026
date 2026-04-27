#!/usr/bin/env python
# coding: utf-8

# <span style="color:blue">
# 
# # Data Treatment Pipeline
# 
# <span>

# In[3]:


# On importe les bibliothèques dont on aura besoin
import pandas as pd
import numpy as np


# In[4]:


# Pipeline Treatment
def pipeline_processing(data):
    data["Embarked"] = data["Embarked"].fillna("S")
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())  # One missing value of Fare in the test dataset
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data = data.drop(columns=["Cabin"])
    data = data.reset_index(drop=True)
    
    # 0 and 1 for Sex and Embarked columns
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    
    # Age category
    thresholds_age = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    labels_age = [0, 1, 2, 3, 4, 5, 6, 7]
    data["AgeCategory"] = pd.cut(data["Age"], bins=thresholds_age, labels=labels_age, include_lowest=True)
    data = data.drop(columns=["Age"])
    
    # Fare category
    thresholds_fare = [0, 25, 100, 200, 300, 600]
    labels_fare = [0, 1, 2, 3, 4]
    data["FareCategory"] = pd.cut(data["Fare"], bins=thresholds_fare, labels=labels_fare, include_lowest=True)
    data = data.drop(columns=["Fare"])

    data = data.drop(columns=["Ticket"])

    # Extract Title from Name
    data["Title"] = data["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Some title are very rare, let's gather them
    rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", 
                   "Sir", "Jonkheer", "Mlle", "Mme", "Ms"]
    
    data["Title"] = data["Title"].replace(rare_titles, "Rare")
    
    # Replacing str by values
    data["Title"] = data["Title"].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4})
    data = data.drop(columns=["Name"])

    # Family is siblings/spouse + parents/children + self
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    
    data = data.drop(columns=["SibSp", "Parch"])

    data.to_csv("Titanic/Titanic_test_processed.csv", index=False)


# In[ ]:




