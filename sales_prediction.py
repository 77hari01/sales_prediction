SALES PREDICTION:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("advertising.csv")
df.head()

df.shape

df.describe()

sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='scatter')
plt.show()

df['TV'].plot.hist(bins=10)

plt.hist(df['Radio'], bins=10, color="red")
plt.xlabel("Radio")
plt.show()

df['Newspaper'].plot.hist(bins=10,color="green",xlabel="newspaper")

sns.heatmap(df.corr(),annot=True)
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df[['TV']],df[['Sales']],test_size=0.3,random_state=0)

X_train

Y_train

X_test

Y_test

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)

res=model.predict(X_test)
res

model.coef_

model.intercept_

0.05473199* 69.2+7.14382225

plt.plot(res)

plt.scatter(X_test,Y_test)
plt.plot(X_test,7.14382225+0.05473199*X_test,'r')
plt.show()