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
    'cont_rating'
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
from sklearn.metrics import r2_score, mean_squared_error


