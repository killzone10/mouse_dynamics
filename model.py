from sklearn.preprocessing import StandardScaler
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from utils.consts import *
import matplotlib.pyplot as plt

class Model():
    def __init__(self, df, users):
        self.df = df
        self.users = users

    def evaluate_sequence_of_samples(self, model, X_validation, y_validation, num_actions):
        ## SINGLE ACTION ## 
        if num_actions == 1: 
            y_scores = model.predict_proba(X_validation)
            return roc_curve(y_validation, y_scores[:, 1])
        ## MANY ACTIONS ##
        X_positive = []
        X_negative = []
        ## ADD TRUE LABELS ##
        for i in range(len(y_validation)):
            if y_validation[i] == 1:
                X_positive.append(X_validation[i])
            ## ADD 0 ##
            else:
                X_negative.append(X_validation[i])

        ## CALCULATE PROBABILITY ##
        pos_scores = model.predict_proba(X_positive)  
        neg_scores = model.predict_proba(X_negative)
        scores =[]
        labels =[]

        n_pos = len(X_positive)
        for i in range(n_pos-num_actions+1): ## num_Actions to wszystkie akcje użytkownika
            score = 0
            for j in range(num_actions):
                score += pos_scores[i + j][1]
            score /= num_actions
            scores.append(score)
            labels.append(1)

        n_neg = len(X_negative)
        for i in range(n_neg - num_actions + 1):
            score = 0
            for j in range(num_actions):
                score += neg_scores[i + j][1]
            score /= num_actions
            scores.append(score)
            labels.append(0)

        return roc_curve(labels, scores)
    

    def scaleData(self, X_train, X_validation):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.fit_transform(X_validation)
            
        return X_train, X_validation
    
    