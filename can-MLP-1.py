#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

def splitdataset(X, Y):
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

    return X_train, X_test, y_train, y_test


# In[ ]:


def preprocess(data, features):

	import pandas as pd
	import numpy as np
	np.random.seed(10)

	N = 3
	print(data.shape)
	row_reduce = data
	for i in range (0, N):
		remove_n = row_reduce.shape[0] // 2
		drop_indices = np.random.choice(row_reduce.index, remove_n, replace=False)
		row_reduce = row_reduce.drop(drop_indices)

	df = row_reduce

	df = df.astype({'DATA[0]': 'str',
					'DATA[1]': 'str',
					'DATA[2]': 'str',
					'DATA[3]': 'str',
					'DATA[4]': 'str',
					'DATA[5]': 'str',
					'DATA[6]': 'str',
					'DATA[7]': 'str',
					'Flag': 'str'})

	print("Preprocessing Y...")
	Y = df.iloc[:, 8]
	le = preprocessing.LabelEncoder()
	Y = le.fit_transform(Y)

	# pd.DataFrame(Y).to_csv("./attack_labels.csv")

	print("Preprocessing X...")
	X = df.iloc[:, 0: 8]
	print(X.shape)


	# LabelEncoder object and fit it to each feature
	print("Encoding X...")
	le = preprocessing.LabelEncoder()
	X = X.apply(le.fit_transform)
	print(X.shape)

	row_reduce = None
	data = None
	df = None

	# OneHotEncoder object, and fit it to all data
	print("One-Hot Encoding X...")
	enc = preprocessing.OneHotEncoder()
	enc.fit(X)
	X = enc.transform(X).toarray()
	print(X.shape)


	from sklearn import datasets, cluster

	print("Performing Feature Agglomeration...")
	agglo = cluster.FeatureAgglomeration(n_clusters = features)
	agglo.fit(X)
	X_reduced = agglo.transform(X)
	print(X_reduced.shape)

	X = None

	return Y, X_reduced


# In[ ]:


df = pd.read_csv('./attacks.csv',index_col=0)
Y, X = preprocess(df, 32)


# In[ ]:


# import numpy as np
# N = 3
# print(df.shape)
# row_reduce = df
# for i in range (0, N):
#     remove_n = row_reduce.shape[0] // 2
#     drop_indices = np.random.choice(row_reduce.index, remove_n, replace=False)
#     row_reduce = row_reduce.drop(drop_indices)

# df = row_reduce
# df.shape


# In[ ]:


# df["Flag"] = df["Flag"].replace(['normal'], 0)
# df["Flag"] = df['Flag'].replace(['DoS'], 1)
# df["Flag"] = df['Flag'].replace(['fuzzy'], 2)
# df["Flag"] = df['Flag'].replace(['gear'], 3)
# df["Flag"] = df['Flag'].replace(['rpm'], 4)


# In[ ]:


# y = df["Flag"]
# X = df.drop(columns = 'Flag')
# df = None
# X = pd.get_dummies(X)


# In[ ]:


X_train, X_test, y_train, y_test = splitdataset(X, Y)


# In[ ]:


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 50), random_state=1)
clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:





# In[ ]:


import numpy as np
from sklearn.model_selection import learning_curve

param_range = [50, 100, 500, 1000, 5000, 10000, 50000, 100000]

train_sizes, train_scores_svm, valid_scores_svm = learning_curve(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 50), random_state=1),
                                                                 X_test, y_test), 
                                                                 train_sizes= param_range, 
                                                                 cv=5)

train_mean_svm = np.mean(train_scores_svm, axis=1)
train_std_svm = np.std(train_scores_svm, axis=1)
test_mean_svm = np.mean(valid_scores_svm, axis=1)
test_std_svm = np.std(valid_scores_svm, axis=1)


# In[ ]:


plt.plot(param_range, train_mean_256, label="Training score (256)", color="red")
plt.plot(param_range, test_mean_256, label="Cross-validation score (256)", color="green")

# Create plot
plt.title("Validation Curve With Decision Tree")
plt.xlabel("Trees")
plt.ylabel("Accuracy Score")
plt.axis([0, 100, 0.9836, 0.984])
plt.legend(loc="best")
plt.show()


# In[ ]:




