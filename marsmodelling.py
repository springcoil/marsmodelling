import pandas as pd
import numpy as np
#from pyearth import Earth
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from pyearth import Earth
from matplotlib import pyplot

df = pd.read_excel('relay-foods.xlsx', sheetname='Purchase Data - Full Study')
df['OrderId'] = df['OrderId'].astype('category')
df['CommonId'] = df['CommonId'].astype('category')


df['OrderId'] = df['OrderId'].astype('category')
df['CommonId'] = df['CommonId'].astype('category')
df.dtypes
col_names = ['OrderDate', 'PickupDate']
df = df.drop(col_names, axis=1)
y = df['TotalCharges']
df_2 = df[['OrderId', 'UserId', 'PupId']]
#del df['OrderDate']
X = [dict(r.iteritems()) for _, r in df_2.iterrows()]
train_fea = DictVectorizer().fit_transform(X)

#Fit an Earth model
model = Earth()
model.fit(train_fea,y)

#Print the model
print(model.trace())
print(model.summary())

#Plot the model
y_hat = model.predict(X)