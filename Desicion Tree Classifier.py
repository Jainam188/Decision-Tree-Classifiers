from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



# CHALLENGE - create 3 more classifiers...
clf = tree.DecisionTreeClassifier()
clf1 = KNeighborsClassifier()
clf2 = RandomForestClassifier()
clf3 = GaussianProcessClassifier()


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 =clf3.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])
prediction1 = clf1.predict([[120, 78, 49]])
prediction2 = clf2.predict([[130,89,33]])
prediction3 = clf3.predict([[150,99,55]])

print(accuracy_score(prediction,prediction3));


# CHALLENGE compare their reusults and print the best one!
print('DecisionTreeClassifier')
print(prediction)
print('KNeighborsClassifier')
print(prediction1)
print('RandomForestClassifier')
print(prediction2)
print('GaussianProcessClassifier')
print(prediction3)