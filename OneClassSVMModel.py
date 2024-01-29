from model import *
from sklearn.svm import OneClassSVM
from sklearn.metrics import recall_score

class OneClassSVMModel(Model):
    def __init__(self, df, users, kernel ='rbf', nu = 0.5, gamma = 0.1 ) :
        super().__init__(df, users)
        if kernel not in {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}:
            raise ValueError("Wront kernel used")
        self.model = OneClassSVM( kernel = kernel, nu = nu, gamma =  gamma)

        self.param_grid = {
            'nu': [0.01, 0.1, 0.2, 0.5],  # Adjust the range based on your dataset
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': [0.1, 0.01, 1,10]  # Adjust the range based on your dataset and kernel choice
        }
    def evaluate(self, X_train, y_train, X_validation, y_validation, user = None, num_actions = 1):
        if user == None:
            print("Dataset with labels was used")
        else:
            print(f"User {user} is being analyzed:")
        X_train, X_validation = self.scaleData(X_train, X_validation)
        self.model.fit(X_train)
        y_predicted = self.model.predict(X_validation)
        test_accuracy = accuracy_score(y_validation, y_predicted)
        print("Test Accuracy: %0.2f" % test_accuracy)
        fpr, tpr, thr = self.evaluate_sequence_of_samples(self.model, X_validation, y_validation, num_actions = num_actions)
        precision, recall, average_precision = self.get_precision_recall_curve(self.model, X_validation, y_validation, num_actions = num_actions)

        return fpr, tpr, thr,  precision, recall, average_precision


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
    



    def get_precision_recall_curve(self, model, X_validation, y_validation, num_actions):
        if num_actions == 1:
            y_scores = model.decision_function(X_validation)
            precision, recall, _ = precision_recall_curve(y_validation, y_scores)
            avg_precision = average_precision_score(y_validation, y_scores)
            return precision, recall, avg_precision

        X_positive = []
        X_negative = []

        for i in range(len(y_validation)):
            if y_validation[i] == 1:
                X_positive.append(X_validation[i])
            else:
                X_negative.append(X_validation[i])

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
            labels.append(0)

        precision, recall, _ = precision_recall_curve(labels, scores)
        avg_precision = average_precision_score(labels, scores)
        return precision, recall, avg_precision
 
    def scorer_f(self, estimator, X, y ):   #your own scorer
        scores = estimator.decision_function(X)  
        auc = roc_auc_score(y, scores)
        return auc
    
    def scorer_f1(self, estimator, X, y):

        scores = estimator.decision_function(X)  
        predictions = np.where(scores >= 0, 1, -1)
        # Calculate recall
        recall = recall_score(y, predictions, average='binary')
        return recall
    
    
    def scorer_f2(self, estimator, X, y):

        scores = estimator.decision_function(X)  
        precision, recall, _ = precision_recall_curve(y, scores)
        
        # Calculate AUC-PR using the trapezoidal rule
        auc_pr = auc(recall, precision)
        return auc_pr
    
    
    def getBestParams(self, X_train, y_train, X_validation, y_validation, user=None, scoring_type='roc_auc', num_actions = 1):
        if user is None:
            print("Dataset with labels was used")
        else:
            print(f"User {user} is being analyzed:")

        X_train, X_validation = self.scaleData(X_train, X_validation)
        estimator = OneClassSVM(nu = 0.5)  # Use 'auto' for now, as it will be tuned during GridSearchCV

        grid_search = GridSearchCV(estimator=estimator, param_grid=self.param_grid, cv= self.kf, scoring = self.scorer_f2)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        # Evaluate the model with the best hyperparameters on the test set
        best_model = grid_search.best_estimator_
        # anomaly_scores = best_model.decision_function(X_validation)
        # print("Anomaly score:", anomaly_scores)
        # predictions = best_model.predict(X_validation)

        # Calculate and print ROC AUC
        fpr, tpr, thresholds = self.evaluate_sequence_of_samples(best_model, X_validation,y_validation, num_actions=num_actions)
        roc_auc = auc(fpr, tpr)
        print("ROC AUC:", roc_auc)
        return best_params, roc_auc
    



   