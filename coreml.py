# -*- coding: utf-8 -*-

import coremltools
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

"""
Importing the digits dataset
Basically we are interested in "images" and "target_names" keys of the dataset
where in "images" key we have n number of image samples represented by <type 'numpy.ndarray'>
Example:

[[  0.   0.   5.  13.   9.   1.   0.   0.]
 [  0.   0.  13.  15.  10.  15.   5.   0.]
 [  0.   3.  15.   2.   0.  11.   8.   0.]
 [  0.   4.  12.   0.   0.   8.   8.   0.]
 [  0.   5.   8.   0.   0.   9.   8.   0.]
 [  0.   4.  11.   0.   1.  12.   7.   0.]
 [  0.   2.  14.   5.  10.  12.   0.   0.]
 [  0.   0.   6.  13.  10.   0.   0.   0.]]

 This particular image represents the digit: 0
"""
digits_dataset = datasets.load_digits()

"""
To apply a classifier on this data, we need to flatten the image, to
turn the data in a (samples, feature) matrix. What this means is that
the matrix above now become this flat array:
[  0.   0.   5.  13.   9.   1.   0.   0.   0.   0.  13.  15.  10.  15.   5.
   0.   0.   3.  15.   2.   0.  11.   8.   0.   0.   4.  12.   0.   0.   8.
   8.   0.   0.   5.   8.   0.   0.   9.   8.   0.   0.   4.  11.   0.   1.
  12.   7.   0.   0.   2.  14.   5.  10.  12.   0.   0.   0.   0.   6.  13.
  10.   0.   0.   0.]
"""
n_samples = len(digits_dataset.images)
data = digits_dataset.images.reshape((n_samples, -1))

# Creating the classifier
classifier = svm.SVC(gamma=0.001)

"""
Now we are splitting training and testing data.
X_train means the array of train images
y_train means the digit class of those images like 0, 1, 2...
X_test means the array of test images
y_test means the digit class of those images like 0, 1, 2...
"""
X = data
y = digits_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Now our model is "learning"
classifier.fit(X_train, y_train)

# Generating feature names as such: feature_0, feature_1 ... feature_63
feature_names = ["feature_"+str(i) for i, x in enumerate(X_train[0])]

# Now it's time to convert it to .mlmodel format
coreml_model = coremltools.converters.sklearn.convert(classifier, feature_names, "digit")
coreml_model.save("Digits.mlmodel")