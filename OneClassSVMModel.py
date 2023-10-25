from model import *
from sklearn.svm import OneClassSVM

class OneClassSVMModel(Model):
    def __init__(self, df, users, kernel ='rbf', nu = 0.5) :
        super().__init__(df, users)
        if kernel not in {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}:
            raise ValueError("Wront kernel used")
        self.model = OneClassSVM(kernel=kernel, nu=nu)

    
    def evaluate(self, X_train, y_train, X_validation, y_validation, user = None):
        if user == None:
            print("Dataset with labels was used")
        else:
            print(f"User {user} is being analyzed:")
        X_train, X_validation = self.scaleData(X_train, X_validation)
        self.model.fit(X_train)
        y_predicted = self.model.predict(X_validation)
        test_accuracy = accuracy_score(y_validation, y_predicted)
        print("Test Accuracy: %0.2f" % test_accuracy)
        fpr, tpr, thr = self.evaluate_sequence_of_samples(self.model, X_validation, y_validation, num_actions = int(self.df.shape[1]/len(self.users)))
        return fpr, tpr, thr


    def evaluate(self, X_train, X_validation):
        X_train, X_validation = self.scaleData(X_train, X_validation)

        self.model.fit(X_train)
        
        # Predict inliers (+1) and outliers (-1)
        y_inliers = np.ones(len(X_validation))
        y_outliers = -np.ones(len(X_validation))
        
        y_predicted = self.model.predict(X_validation)
        
        # Calculate ROC curve and AUC
        fpr, tpr, thr = roc_curve(np.hstack((y_inliers, y_outliers)), np.hstack((y_predicted, y_predicted)))
        roc_auc = auc(fpr, tpr)
        
        # Compute Equal Error Rate (EER)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        threshold = interp1d(fpr, thr)(eer)
        
        print("ROC AUC: %0.2f" % roc_auc)
        print("Equal Error Rate (EER): %0.2f" % eer)
        print("Threshold: %0.2f" % threshold)

        return fpr, tpr, thr