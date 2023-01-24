import MC_Perceptron as mcp
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = shuffle(df)
print(df.tail())

y_train = df.iloc[0:150, 4].values
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
print(le.classes_)
print(y_train)

X_train = df.iloc[0:150, :4].values
print(X_train)

ppnmc = mcp.MC_Perceptron(eta=0.01, n_iter=50)

ppnmc.fit(X_train, y_train)

X_test = df.iloc[0:20, :4].values
y_test = df.iloc[0:20, 4].values
y_test = le.transform(y_test)
print(ppnmc.predict(X_test))
print(y_test)

"""fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(X_train[:, 0][y_train == 0], X_train[:, 1][y_train == 0], X_train[:, 3][y_train == 0], 'r^')
ax.plot3D(X_train[:, 0][y_train == 1], X_train[:, 1][y_train == 1], X_train[:, 3][y_train == 1], 'bs')
ax.plot3D(X_train[:, 0][y_train == 2], X_train[:, 1][y_train == 2], X_train[:, 3][y_train == 2], 'g.')

plt.title('Random Classification Data with 3 classes')
plt.show()"""