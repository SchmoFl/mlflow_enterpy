#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mlflow import tracking
import mlflow
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


tips = pd.read_csv("Daten/Tips_seaborn.csv")
#tips = sns.load_dataset("Tips")

tips = pd.get_dummies(data=tips, columns = ["sex", "smoker", "day", "time"], 
               drop_first=True)

X = tips[tips.columns.difference(["tip"])]
y = tips["tip"]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 42)

dec_tree = DecisionTreeRegressor(min_samples_leaf=20, 
                                     max_depth=4)
    
    
dec_tree.fit(X_train, y_train)


# In[ ]:


dec_tree


# ## Modell Speichern

# In[ ]:


mlflow.sklearn.save_model(dec_tree, 'dec_tree_model')
# mlflow models serve -m dec_tree_model --port 1234 --no-conda


# ## Modell laden

# In[ ]:


mlflow.sklearn.load_model('dec_tree_model')


# In[ ]:


mlflow.pyfunc.load_model('dec_tree_model')


# ## Prediction durch REST-Endpunkt

# In[ ]:


import requests
host = 'localhost'
port = '1234'
url = f'http://{host}:{port}/invocations'
headers = {'Content-Type': 'application/json',}
# test contains our data from the original train/valid/test split
http_data = X_test.head(5).to_json(orient='split')
r = requests.post(url=url, headers=headers, data=http_data)
print(f'Predictions: {r.text}')


# ## Vergleich zur Prediction in python

# In[ ]:


dec_tree.predict(X_test.head(5))


# In[ ]:


y_test.head(2)


# ## Praxisphase IV
# 
# Trainiere ein ML-Modell auf einem beliebigen Datensatz (Trainings-/Testsplit) mit sklearn und speichere es unter einem beliebigen Pfad ab. Stell sicher, dass du das Modell mithilfe des Pfads laden kannst. Benutze zur Vorhersage der ersten X Testdaten:
# 
# * Das via mlflow als REST-API zur Verfügung gestellte Modell
# * Die sklearn-predict-Methode
