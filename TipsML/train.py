import mlflow
import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

data_path = str(sys.argv[2]) if len(sys.argv) > 1 else "Daten/Tips_seaborn.csv"

tips = pd.read_csv(data_path)

tips = pd.get_dummies(data=tips, 
                      columns = ["sex", "smoker", "day", "time"], 
               drop_first=True)

X = tips[tips.columns.difference(["tip"])]
y = tips["tip"]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 42)
                                                    
with mlflow.start_run():
    max_tree_depth = float(sys.argv[1]) if len(sys.argv) > 1 else 4
    min_samp_leaf = 20
    
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