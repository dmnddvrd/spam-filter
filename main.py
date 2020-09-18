import math
from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score

import dataset


class Model:
    def __init__(self, spams=None, hams=None, alpha=0.1):
        self.spams = spams or []
        self.hams = hams or []
        self.alpha = alpha

    def fit(self, X, y):
        for doc, spam in zip(X, y):
            if spam:
                self.spams.append(doc)
            else:
                self.hams.append(doc)
        return self

    def get_params(self, deep=True):
        return {'alpha': self.alpha}

    def set_params(self, alpha):
        self.alpha = alpha
        return self

    def clone(self):
        return Model(self.spams.copy(), self.hams.copy(), self.alpha)

    def predict_values(self, T):
        # min_prob = 0.0000001
        spam = Counter(x for words in self.spams for x in words)
        ham = Counter(x for words in self.hams for x in words)
        p_spam = len(self.spams) / (len(self.spams) + len(self.hams))
        p_ham = 1.0 - p_spam

        sum_spam = sum(spam.values())
        sum_ham = sum(ham.values())

        def is_spam(word):
            if spam[word] == 0:
                # return min_prob
                return (spam[word] + self.alpha) / (self.alpha * len(spam) + sum_spam)
            return spam[word] / sum_spam

        def is_ham(word):
            if ham[word] == 0:
                # return min_prob
                return (ham[word] + self.alpha) / (self.alpha * len(ham) + sum_ham)
            return ham[word] / sum_ham

        def pred(doc):
            res = math.log(p_spam) - math.log(p_ham)
            res += sum(math.log(is_spam(x)) - math.log(is_ham(x)) for x in doc)
            return res

        return [pred(doc) for doc in T]

    def predict(self, T):
        return [x > 0.0 for x in self.predict_values(T)]


X_train, y_train, X_test, y_test = dataset.load()

model = Model()
model.fit(X_train, y_train)

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f'cross_val_score = {scores.mean():.3f} +- {scores.std():.3f}')

param_grid = [
  {'alpha': [1.0, 0.1, 0.01, 0.001, 0.0001]},
]

gs = GridSearchCV(model, param_grid, cv=5, scoring='accuracy').fit(X_train, y_train)
print(f'best_params = {gs.best_params_}')

test_score = accuracy_score(y_test, model.predict(X_test))
print(f'test_score = {test_score:.3f}')

model = gs.best_estimator_

X_ssl = dataset.ssl()
y_ssl = model.predict_values(X_ssl)

X_ssl = [x for x, y in zip(X_ssl, y_ssl) if y > 5.0]
y_ssl = [y for x, y in zip(X_ssl, y_ssl) if y > 5.0]

model.fit(X_ssl, y_ssl)

test_score = accuracy_score(y_test, model.predict(X_test))
print(f'test_score = {test_score:.3f}')
