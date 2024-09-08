from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRFRegressor
import matplotlib.pyplot as plt


#1. 데이터
# x, y = load_iris(return_X_y=True)
# print(x.shape, y.shape)
# (150, 4) (150,)

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


random_state = 1223
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state)

#2. 모델 구성

model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRFRegressor(random_state=random_state,)

models = [model1, model2, model3, model4]

print('random_state :', random_state)
for model in models:
    model.fit(x_train, y_train)
    print('===========', model.__class__.__name__, '==============')
    print('acc', model.score(x_test, y_test))
    print(model.feature_importances_)

import numpy as np

# def plot_feature_importances_dataset(model):
#     n_features =datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#              align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
#     plt.title(model.__class__.__name__)

# plot_feature_importances_dataset(model)
# plt.show()

from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()