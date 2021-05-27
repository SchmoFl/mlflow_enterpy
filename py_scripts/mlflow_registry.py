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


# # mlflow registry

# # Mithilfe der UI

# In[ ]:


tips = pd.read_csv("Daten/Tips_seaborn.csv")

tips = pd.get_dummies(data=tips, columns = ["sex", "smoker", "day", "time"], 
               drop_first=True)

X = tips[tips.columns.difference(["tip"])]
y = tips["tip"]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 42)


# In[ ]:


# mlflow ui -h 0.0.0.0 --backend-store-ui sqlite:///mlflow.db


# In[ ]:


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("http://localhost:5000")


# In[ ]:


with mlflow.start_run(experiment_id="1"):
    max_tree_depth = 4
    min_samp_leaf = 20
    
    mlflow.log_param("max_depth", max_tree_depth)
    mlflow.log_param("min_samples_leaf", min_samp_leaf)
    mlflow.set_tag("model", "dec_tree")
    dec_tree = DecisionTreeRegressor(min_samples_leaf=min_samp_leaf, 
                                     max_depth=max_tree_depth)
    
    
    dec_tree.fit(X_train, y_train)
    tree_predictions = dec_tree.predict(X_test)
    mae = mean_absolute_error(y_test, tree_predictions)
    rmse = mean_squared_error(y_test, tree_predictions, squared=False)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.sklearn.log_model(dec_tree, "sk_tree_model")


# In[ ]:


# export MLFLOW_TRACKING_URI=http://localhost:5000
# mlflow models serve -m "models:/my_first_tree/Production" --no-conda -p 1234


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


# ## Praxisphase Va
# 
# Starte einen mlflow-server/-ui mit einem DB-Backend und setze die tracking-/artifact-uri in deiner python-Session. Starte anschließend einen neuen experiment-run und logge dabei ein Modell. Registriere das Modell über die UI. 

# # Mithilfe der API
# ## Mit der *log_model*-Funktion

# In[ ]:


with mlflow.start_run(experiment_id="1"):
    max_tree_depth = 4
    min_samp_leaf = 20
    
    mlflow.log_param("max_depth", max_tree_depth)
    mlflow.log_param("min_samples_leaf", min_samp_leaf)
    mlflow.set_tag("model", "dec_tree")
    dec_tree = DecisionTreeRegressor(min_samples_leaf=min_samp_leaf, 
                                     max_depth=max_tree_depth)
    
    
    dec_tree.fit(X_train, y_train)
    tree_predictions = dec_tree.predict(X_test)
    mae = mean_absolute_error(y_test, tree_predictions)
    rmse = mean_squared_error(y_test, tree_predictions, squared=False)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)

    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=dec_tree,
        artifact_path="sk_tree_model",
        registered_model_name="my_second_tree"
    )


# ## Mit der *register_model*-Funktion

# In[ ]:


with mlflow.start_run(experiment_id="1"):
    max_tree_depth = 4
    min_samp_leaf = 20
    
    mlflow.log_param("max_depth", max_tree_depth)
    mlflow.log_param("min_samples_leaf", min_samp_leaf)
    mlflow.set_tag("model", "dec_tree")
    dec_tree = DecisionTreeRegressor(min_samples_leaf=min_samp_leaf, 
                                     max_depth=max_tree_depth)
    
    
    dec_tree.fit(X_train, y_train)
    tree_predictions = dec_tree.predict(X_test)
    mae = mean_absolute_error(y_test, tree_predictions)
    rmse = mean_squared_error(y_test, tree_predictions, squared=False)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.sklearn.log_model(dec_tree, "sk_tree_model")


# In[ ]:


result = mlflow.register_model(
    "runs:/717a84cb36c34c0b8197617220c107f5/sk_tree_model",
    "my_third_tree"
)


# ## Mit einem lokalen Client

# In[ ]:


from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")


# In[ ]:


with mlflow.start_run(experiment_id="1"):
    max_tree_depth = 5
    min_samp_leaf = 15
    
    mlflow.log_param("max_depth", max_tree_depth)
    mlflow.log_param("min_samples_leaf", min_samp_leaf)
    mlflow.set_tag("model", "dec_tree")
    dec_tree = DecisionTreeRegressor(min_samples_leaf=min_samp_leaf, 
                                     max_depth=max_tree_depth)
    
    
    dec_tree.fit(X_train, y_train)
    tree_predictions = dec_tree.predict(X_test)
    mae = mean_absolute_error(y_test, tree_predictions)
    rmse = mean_squared_error(y_test, tree_predictions, squared=False)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.sklearn.log_model(dec_tree, "sk_tree_model")


# In[ ]:


client.create_registered_model("empty_model")


# In[ ]:


result = client.create_model_version(
    name="empty_model",
    source="mlruns/1/269f8cf9cdb54d92a3e4b71e15ae7e3f/artifacts/sk_tree_model",
    run_id="269f8cf9cdb54d92a3e4b71e15ae7e3f"
)


# In[ ]:


client.rename_registered_model(
    name="empty_model",
    new_name="non_empty_model"
)


# In[ ]:


client.transition_model_version_stage(
    name="non_empty_model",
    version=1,
    stage="Staging"
)


# In[ ]:


client.delete_registered_model(name="non_empty_model")


# ## Praxisphase Vb
# 
# Starte einen mlflow-server/-ui mit einem DB-Backend und setze die tracking-/artifact-uri in deiner python-Session. Logge und registriere ein Modell auf eine der drei gezeigten Weisen.
