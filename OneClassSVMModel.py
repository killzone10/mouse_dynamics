from model import *
from sklearn.svm import OneClassSVM

class OneClassSVMModel(Model):
    def __init__(self, df, users, kernel ='rbf', nu = 0.5, degree = 3 ) :
        super().__init__(df, users)
        if kernel not in {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}:
            raise ValueError("Wront kernel used")
        self.model = OneClassSVM(kernel = kernel, nu = nu, degree = degree)

    
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


    def evaluate1(self, X_train, X_validation):
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


    def evaluate_sequence_of_samples(self, model, X_validation, y_validation, num_actions):
        ## SINGLE ACTION GIVING ONLY 1 VECTOR
        if num_actions == 1: 
            y_scores = model.decision_function(X_validation)
            return roc_curve(y_validation, y_scores)

        ## MANY ACTIONS MANY VECTORS
        X_positive = []
        X_negative = []
        ## ADD TRUE LABELS (1 for inliers and -1 for outliers)
        for i in range(len(y_validation)):
            if y_validation[i] == 1:
                X_positive.append(X_validation[i])
            ## Change 0 to -1
            else:
                y_validation[i] = -1
                X_negative.append(X_validation[i])

        ## CALCULATE DECISION SCORES
        pos_scores = model.decision_function(X_positive)
        neg_scores = model.decision_function(X_negative)
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