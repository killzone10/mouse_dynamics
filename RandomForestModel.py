from model import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class RandomForestModel(Model):
    def __init__(self, df, users, weight = None, n_estimators = 200, max_depth = 20):
        super().__init__(df, users)
        self.model = RandomForestClassifier(random_state = RANDOM_STATE, class_weight = weight, n_estimators = n_estimators, max_depth = max_depth)

        # self.param_grid = {
        #     'n_estimators': [100, 200, 300],  # Number of trees in the forest
        #     'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
        #     'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
        #     'min_samples_leaf': [1, 2, 4],  # Minimum samples required in a leaf node
        #     'max_features': ['auto', 'sqrt', 'log2'],  # The number of features to consider when looking for the best split
        #     'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
        # }
        self.param_grid = {
            'n_estimators': [100, 200, 300],  # Number of trees in the forest
            'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
            'max_features': ['auto', 'sqrt', 'log2']
        }


    def evaluate(self, X_train, y_train, X_validation, y_validation, scale = True, user = None, num_actions = None):
        if user == None:
            print("Dataset with labels was used")
        else:
            print(f"User {user} is being analyzed:")
        if scale == True:
            X_train, X_validation = self.scaleData(X_train, X_validation)
        self.model.fit(X_train, y_train)
        scores = cross_validate(self.model, X_train, y_train, cv=10, return_train_score=False)
        cv_accuracy = scores['test_score']
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (cv_accuracy.mean(), cv_accuracy.std() * 2))

        y_predicted = self.model.predict(X_validation)
        test_accuracy = accuracy_score(y_validation, y_predicted)
        print("Test Accuracy: %0.2f" % test_accuracy)
        if num_actions == None:
            num_actions = int(self.df.shape[1]/len(self.users))

        fpr, tpr, thr = self.evaluate_sequence_of_samples(self.model, X_validation, y_validation, num_actions = num_actions)
        precision, recall, average_precision = self.get_precision_recall_curve(self.model, X_validation, y_validation, num_actions = num_actions)

        return fpr, tpr, thr, precision, recall, average_precision


    ## calculating feature importance ##
    def calculateFeatureImportance(self, X_train, y_train, X_validation):
        model = RandomForestClassifier(random_state = RANDOM_STATE)
        # X_train, X_validation = self.scaleData(X_train, X_validation)
        model.fit(X_train, y_train)
        feature_importance = model.feature_importances_
        return feature_importance

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

        self.model.fit(X_train, y_train)

        # Calculate feature importances using permutation importance
        feature_importance = permutation_importance(self.model, X_validation, y_validation, n_repeats=50, random_state = RANDOM_STATE)

        return feature_importance.importances_mean
    

    def calculatePermutationImportance1(self, X_train, y_train, X_validation, y_validation):
        # X_train, X_validation = self.scaleData(X_train, X_validation)


        # Calculate feature importances using permutation importance
        feature_importance = permutation_importance(self.model, X_validation, y_validation, n_repeats=50, random_state = RANDOM_STATE)

        return feature_importance.importances_mean
    
    def calculatePermutationImportance2(self, X_train, y_train, X_validation, y_validation):
        # X_train, X_validation = self.scaleData(X_train, X_validation)


        # Calculate feature importances using permutation importance
        feature_importance = permutation_importance(self.model, X_train, y_train, n_repeats=50, random_state = RANDOM_STATE)

        return feature_importance.importances_mean
    


        
    def getBestParams(self, X_train, y_train, X_validation, y_validation, weight = None, user = None, scoring_type = 'roc_auc', num_actions = 1):
        if user == None:
            print("Dataset with labels was used")
        else:
            print(f"User {user} is being analyzed:")

        X_train, X_validation = self.scaleData(X_train, X_validation)
        estimator = RandomForestClassifier(random_state = RANDOM_STATE, class_weight= weight)
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

