from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

import pandas as pd
import matplotlib.pyplot as plt

class Analyser():
    def __init__(self, fileName):
        self.fileName = fileName
        self.readCSV()

    def readCSV(self):
        self.df = pd.read_csv(self.fileName)


    def setPath(self, path):
        self.fileName = path
        self.readCSV()


    def countActions(self):
        action_counts = self.df.groupby(['userid', 'type_of_action']).size().unstack(fill_value=0)
        return action_counts
    
    def plotTypeOfActions(self, stacked = True):
        action_counts = self.countActions()
        action_legend = {1: "MM", 3: "PC", 4: "DD"}
        action_counts.columns = action_counts.columns.map(action_legend)
        action_counts.plot(kind='bar', stacked = stacked)
        plt.title('Type of Actions by User')
        plt.xlabel('User ID')
        plt.ylabel('Count of Actions')
        plt.legend(title='Action Type', loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.show()


    def plotActionHistograms(self):
        # Define a custom legend mapping for action types
        action_legend = {1: "MM", 3: "PC", 4: "DD"}
        action_counts = self.countActions()

        # Map the action types to their corresponding labels
        action_counts.columns = action_counts.columns.map(action_legend)

        # Get the list of unique users
        usersPlot = action_counts.index

        # Determine the number of users
        num_users = len(usersPlot)

        # Set the number of columns for subplots (e.g., 3 columns)
        num_columns = 3

        # Calculate the number of rows required for subplots
        num_rows = (num_users + num_columns - 1) // num_columns

        # Create a single figure with subplots
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 10))

        for i, user in enumerate(usersPlot):
            row = i // num_columns
            col = i % num_columns
            user_data = action_counts.loc[[user]]
            user_data.plot(kind='bar', stacked=False, ax=axes[row, col])
            axes[row, col].set_title(f'User {user}')
            axes[row, col].set_xlabel('Action Type')
            axes[row, col].set_ylabel('Count of Actions')
            axes[row, col].legend(title='Action Type')

        # Hide any remaining empty subplots
        for i in range(num_users, num_rows * num_columns):
            row = i // num_columns
            col = i % num_columns
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.show()
