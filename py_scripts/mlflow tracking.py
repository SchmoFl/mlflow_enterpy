#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mlflow import tracking
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# # MLflow - Python API

# In[ ]:


client = tracking.MlflowClient()


# In[ ]:


experiment_id = client.create_experiment("test_new")


# In[ ]:


experiments = client.list_experiments()
experiments


# In[ ]:


run = client.create_run(experiments[0].experiment_id)
run


# In[ ]:


client.log_param(run.info.run_id, "param1", "hello world")
client.log_param(run.info.run_id, "param2", "123")
client.set_tag(run.info.run_id, "modelType", "test")
client.log_metric(run.info.run_id, "metric1", 30)


# In[ ]:


client.set_terminated(run.info.run_id)


# In[ ]:


client.get_run(run_id)


# In[ ]:


print_run_infos(mlflow.list_run_infos("1"))


# # MLflow - tracking

# In[ ]:


import mlflow


# In[ ]:


tips = pd.read_csv("Daten/Tips_seaborn.csv")

tips = pd.get_dummies(data=tips, columns = ["sex", "smoker", "day", "time"], 
               drop_first=True)

X = tips[tips.columns.difference(["tip"])]
y = tips["tip"]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 42)


# ## Tracking - Params and Metrics

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


# ## Tracking - Artifacts and Models

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
    mlflow.sklearn.log_model(dec_tree, "sk_models")


# In[ ]:


logged_model = 'file:///home/pi/Documents/mlruns/1/c54d2150dca942da815a057c59fc41f5/artifacts/sk_models'

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

loaded_model.predict(X_test)


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
    
    dot_data = export_graphviz(dec_tree, 
                               out_file=None, 
                               feature_names=X_train.columns,
                               label="all", 
                               leaves_parallel=True, 
                               impurity=False, 
                               filled=True)

    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.view(filename="dec_tree")
    mlflow.log_artifact("dec_tree.png")


# ## Praxisphase III
# 
# Trainiere ein beliebiges sklearn-Modell (z.B. Decision/Regression Tree) an einem anderen Datensatz, den du zuvor in Trainings- und Testdaten unterteilst. 
# Trainiere das Modell innerhalb eines runs eines mlflow-Experiments (entweder in einem neuen oder einem bestehenden). Logge dabei die von dir gesetzten 
# Hyperparameter sowie beliebige Metriken und ein Tag, das die Modellklasse beschreibt. Logge in einem zweiten run das Modell selbst und/oder eine 
# Visualisierung oder eine beliebige Textdatei.

# ## Tracking - autolog

# In[ ]:


mlflow.sklearn.autolog(log_input_examples=False, 
                       log_model_signatures=True, 
                       log_models=False, 
                       disable=False, 
                       exclusive=False, 
                       disable_for_unsupported_versions=False, 
                       silent=False)


# In[ ]:


param_grid = {"min_samples_leaf": list(np.linspace(20, 70, 6, dtype = "int64")),
              "min_samples_split": list(np.linspace(50, 80, 4, dtype="int64")),
              "min_impurity_decrease": [0.05, 0.1, 0.15, 0.2]}

dec_tree = DecisionTreeRegressor()

gs_tree = GridSearchCV(dec_tree, param_grid, cv = 5)

with mlflow.start_run(experiment_id="4") as run:
    gs_tree.fit(X_train, y_train)
    tree_predictions = gs_tree.predict(X_test)
    mae = mean_absolute_error(y_test, tree_predictions)
    rmse = mean_squared_error(y_test, tree_predictions, squared=False)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)


# In[ ]:


mlflow.sklearn.autolog(disable=True)


# ## Praxisphase IIIb
# 
# Passe deinen Workflow aus III so an, dass du via Cross-Validation die beste Hyperparameterkombination aus einem von dir definierten 
# Parameter-Grid findest. Benutze die autolog-Methode, um die einzelnen runs der Grid-Search innerhalb eines übergeordneten runs zu tracken.
