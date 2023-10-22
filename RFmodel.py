from model import *
from sklearn.ensemble import RandomForestClassifier

class RFModel(Model):
    def __init__(self, df, users):
        super().__init__(df, users)
        self.model = RandomForestClassifier(random_state = RANDOM_STATE)


    def evaluate(self, X_train, y_train, X_validation, y_validation, user):
        print(f"User {user} is being analyzed:")
        self.model.fit(X_train, y_train)
        scores = cross_validate(self.model, X_train, y_train, cv=10, return_train_score=False)
        cv_accuracy = scores['test_score']
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (cv_accuracy.mean(), cv_accuracy.std() * 2))

        y_predicted = self.model.predict(X_validation)
        test_accuracy = accuracy_score(y_validation, y_predicted)
        print("Test Accuracy: %0.2f" % test_accuracy)
        fpr, tpr, thr = self.evaluate_sequence_of_samples(self.model, X_validation, y_validation, num_actions = int(self.df.shape[1]/len(self.users)))
        return fpr, tpr, thr