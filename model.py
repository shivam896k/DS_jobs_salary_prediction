from operator import index
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv("./analysed.csv")

df.columns

df_model = df[
    [
        "Rating",
        "Size",
        "Type of ownership",
        "Industry",
        "Sector",
        "Revenue",
        "hourly",
        "employer_provided",
        "min_salary",
        "max_salary",
        "avg_salary",
        "job_state",
        "same_state",
        "age",
        "simple_title",
        "seniority",
        "num_competitors",
    ]
]

df_dummy = pd.get_dummies(df_model)

from sklearn.model_selection import train_test_split, cross_val_score
y = df_dummy.avg_salary
X = df_dummy.drop(['avg_salary'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Linear regression
from sklearn.linear_model import LinearRegression, Lasso
lr = LinearRegression()
lr.fit(X_train, y_train)

np.mean(cross_val_score(lr, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))


# Lasso regression
lasso = Lasso(alpha=0.05)
lasso.fit(X_train, y_train)
np.mean(cross_val_score(lasso, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))


alpha = []
error = []

for i in range(1, 101):
    alpha.append(i/100)
    lasso = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lasso, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))


plt.plot(alpha, error)

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns=['alpha', 'error'])

df_err[df_err.error == max(df_err['error'])]




# Random Forest regressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

# GridseaarchCV for tuning
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': range(10, 100, 10), 'criterion': ('mse', 'mae'), 'max_features': ('auto', 'sqrt', 'log2')}

gs_result = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gs_result.fit(X_train, y_train)

gs_result.best_score_
gs_result.best_estimator_


# Predict the output
predict_lr = lasso.predict(X_test)
predict_gs = gs_result.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(predict_lr, y_test)
mean_absolute_error(predict_gs, y_test)






