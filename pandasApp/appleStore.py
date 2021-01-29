import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("./AppleStore.csv")
print(data.head())

num_cols = [
    'size_bytes',
    'price',
    'rating_count_tot',
    'rating_count_ver',
    'sup_devices.num',
    'ipadSc_urls.num',
    'lang.num',
    'cont_rating',
]

cat_cols = [
    'currency',
    'prime_genre'
]

target_col = 'user_rating'

cols = num_cols + cat_cols + [target_col]

data = data[cols]
data['cont_rating'] = data['cont_rating'].str.slice(0, -1).astype(int)

print(data.isna().mean())

for col in cat_cols:
    print(data[col].value_counts())
    print()

data = data.drop(columns=['currency'])
cat_cols.remove('currency')
data.hist(column=num_cols + cat_cols + [target_col], figsize=(14, 10))

pd.plotting.scatter_matrix(data, c=data[target_col], figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=10, alpha=.8)

data['is_free'] = data['price'] == 0
cat_cols.append('is_free')
print(data.head())

data = pd.get_dummies(data, columns=cat_cols)

cat_cols_new = []
for col_name in cat_cols:
    cat_cols_new.extend(filter(lambda x: x.startswith(col_name), data.columns))

from sklearn.preprocessing import StandardScaler

pca = StandardScaler()
pca.fit(data[num_cols + cat_cols])
X = pca.transform(data[num_cols + cat_cols])
# ИЛИ
X = pca.fit_transform(data[num_cols + cat_cols])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, data[target_col], test_size=0.2)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer


def print_metrics(y_preds, y):
    print(f'R^2: {r2_score(y_preds, y)}')
    print(f'MSE: {mean_squared_error(y_preds, y)}')


lr = LinearRegression()
lr.fit(X_train, y_train)

print_metrics(lr.predict(X_test), y_test)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

print_metrics(knn.predict(X_test), y_test)

from sklearn.model_selection import cross_validate

cross_validate(LinearRegression(), X, data[target_col], cv=5,
               scoring={'r2_score': make_scorer(r2_score),
                        'mean_squared_error': make_scorer(mean_squared_error)})

cross_validate(KNeighborsRegressor(), X, data[target_col], cv=5,
               scoring={'r2_score': make_scorer(r2_score),
                        'mean_squared_error': make_scorer(mean_squared_error)})

# искать гиперпараметры

from sklearn.model_selection import GridSearchCV

gbr_grid_search = GridSearchCV(KNeighborsRegressor(),
                               [{'n_neighbors': [1, 2, 3, 4, 6, 8, 10, 15]}],
                               cv=5,
                               error_score=make_scorer(mean_squared_error),
                               verbose=10)
gbr_grid_search.fit(X_train, y_train)

print(gbr_grid_search.best_params_)
print(gbr_grid_search.best_score_)
print(gbr_grid_search.best_estimator_)


