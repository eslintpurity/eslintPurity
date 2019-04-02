#Simple Linear Regression
import matplotlib.pyplot as plt
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()

import matplotlib.pylab as plt
plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()


from sklearn.linear_model import LinearRegression
# Training data
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
# Create and fit the model
model = LinearRegression()
model.fit(X, y)
print 'A 12" pizza should cost: $%.2f' % model.predict([12])[0]
A 12" pizza should cost: $13.68


import numpy as np
>>> import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)
quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()
print X_train
print X_train_quadratic
print X_test
print X_test_quadratic
print 'Simple linear regression r-squared', regressor.score(X_test, y_test)
print 'Quadratic regression r-squared', regressor_quadratic.score(X_test_quadratic, y_test)


from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
df = pd.read_csv('wine/winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print 'R-squared:', regressor.score(X_test, y_test)


import pandas as pd
from sklearn. cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
df = pd.read_csv('data/winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
regressor = LinearRegression()
scores = cross_val_score(regressor, X, y, cv=5)
print scores.mean(), scores

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

scikit-learn provides a convenience function for loading the data set. First, we split 
the data into training and testing sets using train_test_split:

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)

Next, we scaled the features using StandardScaler, which we will describe in detail  in the next chapter:

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print 'Cross validation r-squared scores:', scores
print 'Average cross validation r-squared score:', np.mean(scores)
regressor.fit_transform(X_train, y_train)
print 'Test set r-squared score', regressor.score(X_test, y_test)

Finally, we trained the estimator, and evaluated it using cross validation and the test set. The following is the output of the script:
Cross validation r-squared scores: 
    
## DECISION TREES
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


### K-MEANS
import numpy as np 
from sklearn.cluster import KMeans 
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt 
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
X = np.hstack((cluster1, cluster2)).T
X = np.vstack((x, y)).T 
K = range(1, 10) 
meandistortions = [] 
for k in K: 
kmeans = KMeans(n_clusters=k) 

kmeans.fit(X) 
meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]) 
plt.plot(K, meandistortions, 'bx-') 
plt.xlabel('k') 
plt.ylabel('Average distortion') 
plt.title('Selecting k with the Elbow Method') plt.show()


import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
plt.subplot(3, 2, 1)
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X = np.array(zip(x1, x2)).reshape(len(x1), 2)
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Instances')
plt.scatter(x1, x2)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
tests = [2, 3, 4, 5, 8]
subplot_counter = 1
for t in tests:
subplot_counter += 1
plt.subplot(3, 2, subplot_counter)
kmeans_model = KMeans(n_clusters=t).fit(X)
for i, l in enumerate(kmeans_model.labels_):
plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('K = %s, silhouette coefficient = %.03f' % (t, metrics.silhouette_score(X, kmeans_model.labels_, 
metric='euclidean')))
plt.show()

import numpy as np
import mahotas as mh
from mahotas.features import surf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
import glob

all_instance_filenames = []
all_instance_targets = []
for f in glob.glob('cats-and-dogs-img/*.jpg'): target = 1 if 'cat' in f else 0
all_instance_filenames.append(f)
all_instance_targets.append(target)
surf_features = []
counter = 0
for f in all_instance_filenames:
print 'Reading image:', f
image = mh.imread(f, as_grey=True)
surf_features.append(surf.surf(image)[:, 5:])
train_len = int(len(all_instance_filenames) * .60)
X_train_surf_features = np.concatenate(surf_features[:train_len])
X_test_surf_feautres = np.concatenate(surf_features[train_len:])
y_train = all_instance_targets[:train_len]
y_test = all_instance_targets[train_len:]

n_clusters = 300
print 'Clustering', len(X_train_surf_features), 'features'
estimator = MiniBatchKMeans(n_clusters=n_clusters)
estimator.fit_transform(X_train_surf_features)


X_train = []
for instance in surf_features[:train_len]:
clusters = estimator.predict(instance)
features = np.bincount(clusters)
if len(features) < n_clusters:
features = np.append(features, np.zeros((1, n_clusterslen(features))))
X_train.append(features)
X_test = []
for instance in surf_features[train_len:]:
clusters = estimator.predict(instance)
features = np.bincount(clusters)
if len(features) < n_clusters: features = np.append(features, np.zeros((1, n_clusterslen(features))))
X_test.append(features)

Finally, we train a logistic regression classifier on the feature vectors and targets, and assess its precision, recall, and accuracy:

clf = LogisticRegression(C=0.001, penalty='l2')
clf.fit_transform(X_train, y_train)
predictions = clf.predict(X_test)
print classification_report(y_test, predictions)
print 'Precision: ', precision_score(y_test, predictions)
print 'Recall: ', recall_score(y_test, predictions)
print 'Accuracy: ', accuracy_score(y_test, predictions)


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
y = data.target
X = data.data
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

Finally, we assemble and plot the reduced data:
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)):
if y[i] == 0:
red_x.append(reduced_X[i][0])
red_y.append(reduced_X[i][1])
elif y[i] == 1:
blue_x.append(reduced_X[i][0])
blue_y.append(reduced_X[i][1])
else:
green_x.append(reduced_X[i][0])

green_y.append(reduced_X[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()

from os import walk, path
import numpy as np
import mahotas as mh
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
X = []
y = []

for dir_path, dir_names, file_names in walk('data/att-faces/orl_faces'):
for fn in file_names:
if fn[-3:] == 'pgm':
image_filename = path.join(dir_path, fn)
X.append(scale(mh.imread(image_filename, as_grey=True).reshape(10304).astype('float32')))

y.append(dir_path)
X = np.array(X)

We then randomly split the images into training and test sets, and fit the PCAobject  on the training set:

X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA(n_components=150)

We reduce all of the instances to 150 dimensions and train a logistic regression  classifier. The data set contains forty classes; scikit-learn automatically creates 
binary classifiers using the one versus all strategy behind the scenes:

X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

print 'The original dimensions of the training data were', X_
train.shape
>>> print 'The reduced dimensions of the training data are', X_train_
reduced.shape
>>> classifier = LogisticRegression()
>>> accuracies = cross_val_score(classifier, X_train_reduced, y_train)

>>> print 'Cross validation accuracy:', np.mean(accuracies), 
accuracies
>>> classifier.fit(X_train_reduced, y_train)
>>> predictions = classifier.predict(X_test_reduced)
>>> print classification_report(y_test, predictions)



### MNIST Dataset
First, we import the necessary classes:
from sklearn.datasets import fetch_mldata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

if __name__ == '__main__':
data = fetch_mldata('MNIST original', data_home='data/mnist')
X, y = data.data, data.target
X = X/255.0*2 – 1

X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline = Pipeline([
('clf', SVC(kernel='rbf', gamma=0.01, C=100))
])
print X_train.shape
parameters = {
'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
'clf__C': (0.1, 0.3, 1, 3, 10, 30),
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=2, 
verbose=1, scoring='accuracy')
grid_search.fit(X_train[:10000], y_train[:10000])
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
print '\t%s: %r' % (param_name, best_parameters[param_name])
predictions = grid_search.predict(X_test)
print classification_report(y_test, predictions)


import os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import Image
Next we will define a function that resizes images using the Python Image Library:
def resize_and_crop(image, size):
img_ratio = image.size[0] / float(image.size[1])
ratio = size[0] / float(size[1])
if ratio > img_ratio:
image = image.resize((size[0], size[0] * image.size[1] / 
image.size[0]), Image.ANTIALIAS)
image = image.crop((0, 0, 30, 30))
elif ratio < img_ratio:
image = image.resize((size[1] * image.size[0] / image.size[1], 
size[1]), Image.ANTIALIAS)
image = image.crop((0, 0, 30, 30))
else:
image = image.resize((size[0], size[1]), Image.ANTIALIAS)
return image


X = []
y = []
for path, subdirs, files in os.walk('data/English/Img/GoodImg/Bmp/'):
for filename in files:
f = os.path.join(path, filename)
img = Image.open(f).convert('L') # convert to grayscale
img_resized = resize_and_crop(img, (30, 30))
img_resized = np.asarray(img_resized.getdata(), dtype=np.
float64) \
.reshape((img_resized.size[1] * img_resized.size[0], 1))
target = filename[3:filename.index('-')]
X.append(img_resized)
y.append(target)
X = np.array(X)
X = X.reshape(X.shape[:2])
We will then train a support vector classifier with a polynomial 
kernel.classifier = SVC(verbose=0, kernel='poly', degree=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_
state=1)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print classification_report(y_test, predictions)