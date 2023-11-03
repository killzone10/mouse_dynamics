from model import *
from sklearn.ensemble import IsolationForest




class IsolationForestModel(Model):
    def __init__(self, df, users, contamination=0.1, n_estimators=100, random_state=None):
        super().__init__(df, users)
        self.model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=random_state)

    def evaluate(self, X_train, y_train, X_validation, y_validation, user = None):
        if user is None:
            print("Dataset with labels was used")
        else:
            print(f"User {user} is being analyzed:")
        X_train, X_validation = self.scaleData(X_train, X_validation)
        self.model.fit(X_train)
        y_predicted = self.model.predict(X_validation)
        test_accuracy = accuracy_score(y_validation, y_predicted)
        print("Test Accuracy: %0.2f" % test_accuracy)
        fpr, tpr, thr = self.evaluate_sequence_of_samples(self.model, X_validation, y_validation,  num_actions = int(self.df.shape[1] / len(self.users)))
        return fpr, tpr, thr

    def evaluate_sequence_of_samples(self, model, X_validation, y_validation,  num_actions):
        ## SINGLE ACTION GIVING ONLY 1 VECTOR
        if num_actions == 1:
            scores = model.decision_function(X_validation)  # Reverse the decision scores
            return roc_curve(y_validation, scores)

        ## MANY ACTIONS MANY VECTORS
        X_positive = []
        X_negative = []

        ## ADD TRUE LABELS (1 for inliers and -1 for outliers)
        for i in range(len(y_validation)):
            if y_validation[i] == 1:
                X_positive.append(X_validation[i])
            else:
                y_validation[i] = -1
                X_negative.append(X_validation[i])

        ## CALCULATE AVERAGE ANOMALY SCORE
        pos_scores = model.decision_function(X_positive)  # Reverse the decision scores
        neg_scores = model.decision_function(X_negative)  # Reverse the decision scores
        scores = []
        labels = []

        n_pos = len(X_positive)
        for i in range(n_pos - num_actions + 1):
            score = 0
            for j in range(num_actions):
                score += pos_scores[i + j]
            score /= num_actions
            scores.append(score)
            labels.append(1)

        n_neg = len(X_negative)
        for i in range(n_neg - num_actions + 1):
            score = 0
            for j in range(num_actions):
                score += neg_scores[i + j]
            score /= num_actions
            scores.append(score)
            labels.append(-1)

        return roc_curve(labels, scores)
