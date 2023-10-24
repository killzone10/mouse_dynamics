from model import *
from sklearn.svm import SVC

class SVMModel(Model):
    def __init__(self, df, users, kernel ='rbf', C = 0.1, weight = 'balanced') :
        super().__init__(df, users)
        if kernel not in {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}:
            raise ValueError("Wront kernel used")
        
        self.model = SVC(kernel='rbf', class_weight= weight, C = C, probability=True) ## TODO check C coefficient

    
    def evaluate(self, X_train, y_train, X_validation, y_validation, user = None):
        if user == None:
            print("Dataset with labels was used")
        else:
            print(f"User {user} is being analyzed:")
        X_train, X_validation = self.scaleData(X_train, X_validation)
        self.model.fit(X_train, y_train)
        scores = cross_validate(self.model, X_train, y_train, cv=10, return_train_score=False)
        cv_accuracy = scores['test_score'] # LET IT BE
        # cv_accuracy = scores['test_precision_macro']

        print("CV Accuracy: %0.2f (+/- %0.2f)" % (cv_accuracy.mean(), cv_accuracy.std() * 2))

        y_predicted = self.model.predict(X_validation)
        test_accuracy = accuracy_score(y_validation, y_predicted)
        print("Test Accuracy: %0.2f" % test_accuracy)
        fpr, tpr, thr = self.evaluate_sequence_of_samples(self.model, X_validation, y_validation, num_actions = int(self.df.shape[1]/len(self.users)))
        return fpr, tpr, thr


    