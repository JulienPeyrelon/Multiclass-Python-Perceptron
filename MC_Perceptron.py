import numpy as np
import pandas as pd
import Perceptron as p
import re
import itertools


class MC_Perceptron(object):
    """Multiclass Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    """

    def __init__(self, eta=0.1, n_iter=100, strat="ovo"):
        self.ppns_names = None
        self.ppns = None
        self.eta = eta
        self.n_iter = n_iter
        self.strat = strat

    def fit(self, X, y):

        classes = pd.Series(y).unique()
        nbr_of_class = len(classes)

        print("classes", classes)

        if (self.strat == "ova"):
            """1 class vs all other classes"""
            self.ppns = []
            self.ppns_names = []
            y_class = []
            for i in classes:
                string = "y_" + str(i)
                y_class.append(string)
                print(string)
                my_vars = locals()
                my_vars[string] = y
                my_vars[string] = np.where(my_vars[string] == i, -1, 1)

                string2 = "ppn_" + str(i)
                my_vars[string2] = p.Perceptron(eta=0.01, n_iter=10)
                my_vars[string2].fit(X, my_vars[string])

                self.ppns.append(my_vars[string2])
                self.ppns_names.append(string2)
        else:
            """1 class vs one other"""
            self.ppns = []
            self.ppns_names = []
            c = list(set(itertools.combinations(classes, 2)))
            print(c)
            for tup in c:
                string = "y_" + str(tup[0]) + str(tup[1])
                my_vars = locals()
                my_vars[string] = y

                x_temp = np.delete(X,np.where((my_vars[string] != tup[0]) & (my_vars[string] != tup[1]))[0], 0)
                my_vars[string] = np.delete(my_vars[string],
                                            np.where((my_vars[string] != tup[0]) & (my_vars[string] != tup[1]))[0])
                my_vars[string] = np.where(my_vars[string] == tup[0], -1, 1)

                string2 = "ppn_" + str(tup[0]) + str(tup[1])
                my_vars[string2] = p.Perceptron(eta=0.01, n_iter=10)
                my_vars[string2].fit(x_temp, my_vars[string])
                self.ppns.append(my_vars[string2])
                self.ppns_names.append(string2)

        return self

    def predict(self, X):
        """return a poll of all perceptron"""
        y_pred = []

        if (self.strat == "ova"):
            for x in X:
                counter = 0
                pred = True
                for ppn in self.ppns:
                    current_class = re.findall(r'\d+', self.ppns_names[counter])
                    if (ppn.predict(x) == -1 and pred):
                        y_pred.append(int(current_class[0]))
                        pred = False
                    counter += 1
                if (pred):
                    y_pred.append(9)

        else:
            for x in X:
                counter = 0
                poll = []
                for ppn in self.ppns:
                    current_tuple = re.findall(r'\d+', self.ppns_names[counter])
                    if (ppn.predict(x) == -1):
                        poll.append(list(current_tuple[0])[0])
                    else:
                        poll.append(list(current_tuple[0])[1])
                    counter += 1
                #print(poll)
                y_pred.append(int(max(set(poll), key=poll.count)))

        return np.array(y_pred)
