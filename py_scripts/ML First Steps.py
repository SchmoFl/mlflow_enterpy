#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import graphviz


# # Explorative Datenanalyse und Erste Modelle

# In[ ]:


ysnp = pd.read_csv("Daten/ysnp.csv")

ysnp.head(3)


# In[ ]:


ysnp.columns


# In[ ]:


ysnp.shape


# In[ ]:


# Umrechnung von Fahrenheit in Celsius
ysnp.filter(regex = "Temperature")


# In[ ]:


Temp_cols = ysnp.filter(regex = "Temperature").columns

ysnp[Temp_cols] = ysnp[Temp_cols].apply(lambda x: (x - 32)/1.8)

ysnp.columns = list(map(lambda x: x.replace('(F)', '(C)'), 
                        ysnp.columns))


# In[ ]:


ysnp.dtypes


# In[ ]:


ysnp["Year/Month/Day"] = pd.to_datetime(ysnp["Year/Month/Day"], 
    format = '%Y/%m/%d')

ysnp["Year/Month/Day"].agg({"min", "max"})

ysnp["Year"] = ysnp["Year/Month/Day"].apply(lambda x: x.year)
ysnp["Year"].head()

ysnp["Month"] = ysnp["Year/Month/Day"].apply(lambda x: x.month)


# In[ ]:


# Anzahl der Beobachtungen pro Jahr
ysnp.groupby("Year").size()
set(ysnp.groupby("Year").size())


# In[ ]:


# Fehlende Werte
ysnp.info()


# In[ ]:


ysnp = ysnp.dropna()


# In[ ]:


# Visualisierungen
Plot_Data = ysnp[ysnp.columns.difference(["Year/Month/Day", 
                                          "Month", 
                                          "Year", 
                                          "Recreation Visits"])]

sns.boxplot(data = Plot_Data)
plt.xticks(rotation = 90)
plt.tight_layout()


# In[ ]:


sns.lineplot(x = "Year/Month/Day", y = "Recreation Visits", data = ysnp)


# In[ ]:


sns.boxplot(x = "Month", y = "Recreation Visits", data = ysnp)


# In[ ]:


sns.lmplot(x = "MeanTemperature(C)", y = "Recreation Visits", 
           data = ysnp)


# In[ ]:


# Umwandeln des Monats in Dummy-Variablen
pd.get_dummies(data = ysnp["Month"], columns = "Month", 
               drop_first=True)


# ## Lineare Regression

# In[ ]:


ysnp = pd.get_dummies(data = ysnp, columns = ["Month"], 
                      drop_first=True)

# Lineare Regression mit scikit-learn
X = ysnp[["MeanTemperature(C)"]]

y = ysnp["Recreation Visits"]

RegMod = LinearRegression(fit_intercept=True)

RegMod.fit(X, y)


# In[ ]:


RegMod.coef_


# In[ ]:


RegMod.intercept_


# In[ ]:


RegMod.score(X, y)


# In[ ]:


# Berechnung der Residuen (Fehler)
Residuals_model = y - RegMod.predict(X)
Residuals_model

Residuals_Df = pd.DataFrame(Residuals_model).rename(columns = {'Recreation Visits':'Residuen'})

# Visualisierung der Fehlerverteilung --> im Allgemeinen sollten die Fehler normalverteilt (also zufällig) sein
Residuals_Df.plot(kind = "hist", legend = False)


# In[ ]:


# Berechnung einer linearen Regression mit mehr als einem Feature
X = ysnp[ysnp.columns.difference(["Recreation Visits",
                                  "Year/Month/Day",
                                  "Year"])]
y = ysnp["Recreation Visits"]

RegMod_2 = LinearRegression(fit_intercept=True)
RegMod_2.fit(X, y)

RegMod_2.coef_
RegMod_2.score(X, y)


# In[ ]:


pd.DataFrame(y - RegMod_2.predict(X)).plot(kind="hist", legend=False)


# ## Decision/Regression Trees

# In[ ]:


tips = pd.read_csv("Daten/Tips_seaborn.csv")
tips = sns.load_dataset("Tips")

tips.head(3)


# In[ ]:


tips = pd.get_dummies(data=tips, columns = ["sex", "smoker", "day", "time"], 
               drop_first=True)


# In[ ]:


reg_tree = DecisionTreeRegressor(min_samples_leaf=20)

X = tips[tips.columns.difference(["tip"])]

y = tips["tip"]

reg_tree.fit(X,y)

reg_tree.score(X, y)

dot_data = export_graphviz(reg_tree, out_file=None, feature_names=X.columns,
                           label = "all", leaves_parallel = True, 
                           impurity = False, filled=True)

graph = graphviz.Source(dot_data)
graph.format = "png"
graph.view(filename="dec_tree")


# In[ ]:


from IPython.display import Image
Image('dec_tree.png')


# ## Praxisphase I 
# 
# Lade dir einen neuen Datensatz in die Session:
# 
# * Eins der csv-files aus dem git-repo
# * Mithilfe der seaborn-Library, s. Funktion seaborn.load_dataset und https://github.com/mwaskom/seaborn-data für verfügbare Datensätze
# * Einen eigenen Datensatz
# 
# Verschaffe dir einen Überblick, wähle Ziel- und voraussagende Variablen aus und erstelle ein erstes Modell (z.B. DecisionTree).
# 

# # Basic Machine Learning 

# ## Training/Test-Split

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


X = ysnp[ysnp.columns.difference(["Recreation Visits", "Year/Month/Day", "Year"])]
y = ysnp["Recreation Visits"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 42)


# In[ ]:


# Trainiere einen Regression-Tree auf den Trainingsdaten
dec_tree = DecisionTreeRegressor(min_samples_leaf = 20, max_depth = 4)
dec_tree.fit(X_train, y_train)


# Trainiere eine Lineare Regression auf den Trainingsdaten
lin_reg = LinearRegression(fit_intercept = True)
lin_reg.fit(X_train, y_train)

# Wende die Modelle auf die Testdaten an
tree_predictions = dec_tree.predict(X_test)
lin_reg_predictions = lin_reg.predict(X_test)


# In[ ]:


mean_absolute_error(y_test, tree_predictions)


# In[ ]:


mean_absolute_error(y_test, lin_reg_predictions)


# In[ ]:


param_grid = {"min_samples_leaf": list(np.linspace(20, 70, 6, dtype = "int64")),
              "min_samples_split": list(np.linspace(50, 80, 4, dtype="int64")),
              "min_impurity_decrease": [0.05, 0.1, 0.15, 0.2]}

ysnp_tree = DecisionTreeRegressor()

gs_tree = GridSearchCV(ysnp_tree, param_grid, cv = 5)

gs_tree.fit(X_train, y_train)


# In[ ]:


gs_tree.best_params_


# In[ ]:


gs_tree.best_estimator_


# In[ ]:


gs_tree.best_score_


# In[ ]:


preds = gs_tree.predict(X_test)

mean_absolute_error(y_test, preds)


# ## Praxisphase II 
# 
# Wähle einen neuen Datensatz aus und bestimme Zielvariable sowie voraussagende Variablen.
# 
# 1. Erstelle ein Trainings- und Testsplit
# 1. Erstelle ein parameter-grid für einen Decision Tree (oder für ein anderes sklearn-Modell, s. Doku)
# 1. Bestimme die beste Hyperparameterkombination mithilfe einer Cross-Validation
# 1. Wende das Modell auf die Testdaten an und bestimme die Güte anhand einer beliebigen Metrik
# 
