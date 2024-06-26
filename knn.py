import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split



class KNN:

    def __init__(self, k, dist="o"):
        self.k= k
        self.distance_ = dist

    def fit(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

    def predict(self, xtest):

        predict_lables = []

        for x_test in xtest:

            distance_list = []
            for x_train in self.xtrain:
                if self.distance_ == "ch":
                    distance = KNN.chebichof_distance(x_test, x_train)
                elif self.distance_ == "m":
                    distance = KNN.manhatan_distance(x_test, x_train)
                else:
                    distance = KNN.oglidos_distance(x_test, x_train)

                distance_list.append(distance)

            indexes = np.argsort(distance_list)
            k_indexes = indexes[:self.k]

            k_nearest_lables = []
            for i in k_indexes:
                k_nearest_lables.append(self.ytrain[i])
            most_common = Counter(k_nearest_lables).most_common(1)
            predict_lables.append(most_common[0][0])

        self.predict_ = np.array(predict_lables)
        return np.array(predict_lables)

    def accurancy(self, y_test):
        acc = 0
        for i in range(len(self.predict_)):

            if self.predict_[i] == y_test[i]:
                acc = acc + 1

        return (acc / len(predicted)) * 100

    @classmethod
    def oglidos_distance(self,x,y):
        distance = np.sqrt(np.sum(x - y) ** 2)
        return distance

    @classmethod
    def manhatan_distance(clsself, x, y):
        distance = np.sum(np.abs(x - y))
        return distance

    @classmethod
    def chebichof_distance(clsself, x, y):
        distance = np.max(np.abs(x - y))
        return distance



df = pd.read_csv("Social_Network_Ads.csv")
df = df[["Age", "EstimatedSalary", "Purchased"]]
clas_name = "Purchased"

X = np.array(list(zip(df["Age"], df["EstimatedSalary"])))
Y = np.array(df[clas_name])
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.2, random_state=10)


model = KNN(5, "m")
model.fit(x_train, y_train)
predicted = model.predict(x_test)

acc = model.accurancy(y_test)
print(acc)














