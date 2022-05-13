import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

#import data and split
peptide = pd.read_excel(r'imput.xlsx',sheet_name='Sheet1')
print (peptide)
X=peptide.drop(['Activity'], axis=1)
y=peptide['Activity']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
rf = RandomForestRegressor(random_state=0)

rf.fit(X_train,y_train)
f_i = list(zip(features,rf.feature_importances_))
f_i.sort(key = lambda x : x[1])
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])

plt.show()
rfe = RFECV(rf,cv=5,scoring="neg_mean_squared_error")

rfe.fit(X_train,y_train)
selected_features = np.array(features)[rfe.get_support()]