from model import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class RFModel(Model):
    def __init__(self, df, users):
        super().__init__(df, users)
        self.model = RandomForestClassifier(random_state = RANDOM_STATE)


    def evaluate(self, X_train, y_train, X_validation, y_validation, scale = True, user = None):
        if user == None:
            print("Dataset with labels was used")
        else:
            print(f"User {user} is being analyzed:")
        X_train, X_validation = self.scaleData(X_train, X_validation)
        self.model.fit(X_train, y_train)
        scores = cross_validate(self.model, X_train, y_train, cv=10, return_train_score=False)
        cv_accuracy = scores['test_score']
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (cv_accuracy.mean(), cv_accuracy.std() * 2))

        y_predicted = self.model.predict(X_validation)
        test_accuracy = accuracy_score(y_validation, y_predicted)
        print("Test Accuracy: %0.2f" % test_accuracy)
        fpr, tpr, thr = self.evaluate_sequence_of_samples(self.model, X_validation, y_validation, num_actions = int(self.df.shape[1]/len(self.users)))
        return fpr, tpr, thr


    def calculateFeatureImportance(self, X_train, y_train, X_validation):
        sumOfFeatures = np.array(0)
        counter = 0 
        for i in range (1, 100):
            counter +=1 
            model = RandomForestClassifier(random_state = RANDOM_STATE)
            # X_train, X_validation = self.scaleData(X_train, X_validation)
            model.fit(X_train, y_train)
            feature_importance = model.feature_importances_
            sumOfFeatures = np.add(sumOfFeatures, feature_importance)

        return sumOfFeatures/counter


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