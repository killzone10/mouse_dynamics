from model import *
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

class SVMModel(Model):
    def __init__(self, df, users, kernel ='rbf', C = 1, weight = None) :
        super().__init__(df, users)
        if kernel not in {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}:
            raise ValueError("Wront kernel used")
        
        self.param_grid = {
            'C': [0.1, 1, 10],# Regularization parameter
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel types
            'gamma': [0.1, 1, 10], # Kernel coefficient (only for 'rbf' and 'poly' kernels)
        }
        # self.param_grid = {
        #     'C': [0.1],# Regularization parameter
        #     'kernel': [ 'rbf'],  # Kernel types
        #     'gamma': [0.1], # Kernel coefficient (only for 'rbf' and 'poly' kernels)
        # }
        self.model = SVC(kernel='rbf', class_weight = weight, C = C, probability=True) ## TODO check C coefficient

    def evaluate(self, X_train, y_train, X_validation, y_validation, user = None, num_actions = None):
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
        if num_actions == None:
            num_actions = int(self.df.shape[1]/len(self.users))

        fpr, tpr, thr = self.evaluate_sequence_of_samples(self.model, X_validation, y_validation, num_actions = num_actions)
        precision, recall, average_precision = self.get_precision_recall_curve(self.model, X_validation, y_validation, num_actions = num_actions)

        return fpr, tpr, thr, precision, recall, average_precision


    
    def getBestParams(self, X_train, y_train, X_validation, y_validation, weight = None, user = None, scoring_type = 'roc_auc', num_actions = 1):
        if user == None:
            print("Dataset with labels was used")
        else:
            print(f"User {user} is being analyzed:")

        X_train, X_validation = self.scaleData(X_train, X_validation)
        estimator = SVC(class_weight= weight, probability=True)
        if scoring_type == 'roc_auc':
            print("Scoring/ Refit ROC AUC:")
            grid_search = GridSearchCV(estimator=estimator, param_grid = self.param_grid, cv = self.kf, scoring = 'roc_auc', refit = 'roc_auc')
        else:
            print("Scoring/ Refit accuracy")
            grid_search = GridSearchCV(estimator=estimator, param_grid = self.param_grid, cv = self.kf, scoring = 'accuracy', refit = 'accuracy')

        grid_search.fit(X_train, y_train)
        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)
        # Evaluate the model with the best hyperparameters on the test set
        best_model = grid_search.best_estimator_
        accuracy = best_model.score(X_validation, y_validation)
        print("Test Set Accuracy:", accuracy)

          # Calculate and print ROC AUC
        # y_pred = best_model.decision_function(X_validation)  # decision_function for SVC provides the decision values
          # Calculate and print ROC AUC
        fpr, tpr, thr = self.evaluate_sequence_of_samples(best_model, X_validation, y_validation, num_actions = num_actions)
        # fpr, tpr, thresholds = roc_curve(y_validation, best_model.predict_proba(X_validation)[:, 1])

        roc_auc = auc(fpr, tpr)
        print("ROC AUC:", roc_auc)

        return best_params, roc_auc



    ## ploting feature importance ##
    def plotFeatureImportance(self, featureImportance, X, df):
        plt.figure(figsize=(8, 10))
        num_features = X.shape[1]  # Number of features in your NumPy array
        # plt.barh(range(num_features), feature_importance, align='center')
        plt.barh(range(num_features), featureImportance, align='center')

        feature_names = df.columns.tolist()
        plt.yticks(range(num_features), feature_names[:-1])  # Use indices as feature names
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Feature Importances in RandomForestClassifier')
        plt.show()



    ## PERMUTATION IMPORTANCE ## 
    def calculatePermutationImportance(self, X_train, y_train, X_validation, y_validation):
        # X_train, X_validation = self.scaleData(X_train, X_validation)
        X_train, X_validation = self.scaleData(X_train, X_validation)

        self.model.fit(X_train, y_train)

        # Calculate feature importances using permutation importance
        feature_importance = permutation_importance(self.model, X_validation, y_validation, n_repeats=50, random_state = RANDOM_STATE)

        return feature_importance.importances_mean
    

    def plotFeatureImportance(self, featureImportance, X, df):
        plt.figure(figsize=(8, 10))
        num_features = X.shape[1]  # Number of features in your NumPy array
        # plt.barh(range(num_features), feature_importance, align='center')
        plt.barh(range(num_features), featureImportance, align='center')

        feature_names = df.columns.tolist()
        plt.yticks(range(num_features), feature_names[:-1])  # Use indices as feature names
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Feature Importances in SVM Model')
        plt.show()
