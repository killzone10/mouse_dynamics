from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager, freeze_support
import pandas as pd
from RandomForestModel import *
from SVMmodel import *
from OneClassSVMModel import *
from IsolationForestModel import *

from nonlegality_analyser import * ## the class which analyses the datasets without 0 1 labels --> splitting samples is being done there ##  
from utils.consts import * ## const variables are there, so paths and extraction features ##
from utils. plotting import * ## plotting ##
from data_reader_babalit import * ## the class which reads balabit dataset ##
from data_reader_chaoshen import * ## the class which reads chaoshen datasets TODO There are several issues with this dataset -> explained later ##
from legality_analyser import * ## the class which analyses the datasets with 0 1 labels --> splitting samples is being done there ##  
import random
import re
## getting path from the reader ##
path = 'processed_files\\balabit_dataset_users[7, 9, 12, 15, 16, 20, 21, 23, 29, 35]_limit100000000_labelsFalse.csv'
# path = 'processed_files\chaoshen_dataset_users[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]_limit1000_labelsFalse.csv'
# path = 'processed_files\Singapur_dataset_users[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]_limit100000000_labelsFalse.csv'
# path = 'processed_files\dfl_dataset_users[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]_limit100000000_labelsFalse.csv'
# path = ''
## creating analyser class ## 
dflAnalyser = nonLegalityAnalyser(path)
## counting how many actions were extracted ## 


match = re.search(r'_limit(\d+)_', path)

if match:
    extracted_number = int(match.group(1))
    print("Extracted Number:", extracted_number)
else:
    print("Number not found in the path.")

# users = [i for i in range (1,22)] # all users DFL
# users = [i for i in range (1,29)] ## all users  chaoshen
users = BALABIT_USERS
# Assume 'users', 'dflAnalyser', 'TEST_SIZE', 'RandomForestModel', 'auc', 'interp1d', 'brentq', 'plotROCs' are defined elsewhere in your code.

shuffle = True

# Use a multiprocessing Manager to create shared dictionaries
balanced = True

def process_user(legalUser, fpr, tpr, roc_auc, precission, recall, average_precision):
    dataset = dflAnalyser.createTrainingDataWithLabel(legalUser, balanced=balanced)
    X = dataset[:, 0:-1]
    y = dataset[:, -1]
    X_train, X_validation, y_train, y_validation = dflAnalyser.trainingTestSplit(X, y, TEST_SIZE, shuffle)
    if balanced:
        model = RandomForestModel(dataset, users)

    else:
        model = RandomForestModel(dataset, users, weight='balanced')

    fpr_user, tpr_user, thr, precission_user, recall_user, average_precision_user = model.evaluate(X_train, y_train, X_validation, y_validation, scale=True, user=legalUser, num_actions=1)
    threshold = -1
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr_user, tpr_user)(x), 0., 1.)
        threshold = interp1d(fpr_user, thr)(eer)
    except (ZeroDivisionError, ValueError):
        print("Division by zero")

    # Update shared dictionaries
    fpr[legalUser] = fpr_user
    tpr[legalUser] = tpr_user
    roc_auc[legalUser] = auc(fpr_user, tpr_user)
    precission[legalUser] = precission_user
    recall[legalUser] = recall_user
    average_precision[legalUser] = average_precision_user

    print(f"{legalUser}: {roc_auc[legalUser]:.3f} threshold: {threshold:.3f}")
    print(f"{legalUser}: Average Precision {average_precision[legalUser]:.3f} ")

def process_user_svm(legalUser, fpr, tpr, roc_auc, precission, recall, average_precision):
    dataset = dflAnalyser.createTrainingDataWithLabel(legalUser, balanced=balanced)
    X = dataset[:, 0:-1]
    y = dataset[:, -1]
    X_train, X_validation, y_train, y_validation = dflAnalyser.trainingTestSplit(X, y, TEST_SIZE, shuffle)
    if balanced:
        model = SVMModel(dataset, users)
    else:
        model = SVMModel(dataset, users, weight='balanced')
    fpr_user, tpr_user, thr, precission_user, recall_user, average_precision_user = model.evaluate(X_train, y_train, X_validation, y_validation, user=legalUser, num_actions=1)
    threshold = -1
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr_user, tpr_user)(x), 0., 1.)
        threshold = interp1d(fpr_user, thr)(eer)
    except (ZeroDivisionError, ValueError):
        print("Division by zero")

    # Update shared dictionaries
    fpr[legalUser] = fpr_user
    tpr[legalUser] = tpr_user
    roc_auc[legalUser] = auc(fpr_user, tpr_user)
    precission[legalUser] = precission_user
    recall[legalUser] = recall_user
    average_precision[legalUser] = average_precision_user

    print(f"{legalUser}: {roc_auc[legalUser]:.3f} threshold: {threshold:.3f}")
    print(f"{legalUser}: Average Precision {average_precision[legalUser]:.3f} ")    



def process_user_oc_svm(legalUser, fpr, tpr, roc_auc, precission, recall, average_precision):
    dataset = dflAnalyser.createTrainingDataWithLabel(legalUser, balanced=balanced)
 

    X = dataset[:, 0:-1]
    y = dataset[:, -1]
    
    x_negative = X[y == 0]
    x_positive = X[y == 1] 

    X_train, X_validation, y_train, y_validation = dflAnalyser.trainingTestSplit(X, y, TEST_SIZE, shuffle)
    y_validation = np.where(y_validation == 0, -1, y_validation)

    model = OneClassSVMModel(dataset, users, nu = 0.5, kernel="rbf", gamma = 5)
    fpr_user, tpr_user, thr, precission_user, recall_user, average_precision_user = model.evaluate(x_positive, y_train, X_validation, y_validation, legalUser, num_actions=1)
    threshold = -1
 
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr_user, tpr_user)(x), 0., 1.)
        threshold = interp1d(fpr_user, thr)(eer)
    except (ZeroDivisionError, ValueError):
        print("Division by zero")

    # Update shared dictionaries
    fpr[legalUser] = fpr_user
    tpr[legalUser] = tpr_user
    roc_auc[legalUser] = auc(fpr_user, tpr_user)
    precission[legalUser] = precission_user
    recall[legalUser] = recall_user
    average_precision[legalUser] = average_precision_user

    print(f"{legalUser}: {roc_auc[legalUser]:.3f} threshold: {threshold:.3f}")
    print(f"{legalUser}: Average Precision {average_precision[legalUser]:.3f} ")   



def process_user_iso_forest(legalUser, fpr, tpr, roc_auc, precission, recall, average_precision):
    dataset = dflAnalyser.createTrainingDataWithLabel(legalUser, balanced=balanced)
 

    X = dataset[:, 0:-1]
    y = dataset[:, -1]
    
    x_negative = X[y == 0]
    x_positive = X[y == 1] 

    X_train, X_validation, y_train, y_validation = dflAnalyser.trainingTestSplit(X, y, TEST_SIZE, shuffle)
    y_validation = np.where(y_validation == 0, -1, y_validation)

    model = IsolationForestModel(dataset, users, contamination= 0.5, n_estimators=200)
    fpr_user, tpr_user, thr, precission_user, recall_user, average_precision_user = model.evaluate(x_positive, y_train, X_validation, y_validation, legalUser, num_actions=1)
    threshold = -1
 
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr_user, tpr_user)(x), 0., 1.)
        threshold = interp1d(fpr_user, thr)(eer)
    except (ZeroDivisionError, ValueError):
        print("Division by zero")

    # Update shared dictionaries
    fpr[legalUser] = fpr_user
    tpr[legalUser] = tpr_user
    roc_auc[legalUser] = auc(fpr_user, tpr_user)
    precission[legalUser] = precission_user
    recall[legalUser] = recall_user
    average_precision[legalUser] = average_precision_user

    print(f"{legalUser}: {roc_auc[legalUser]:.3f} threshold: {threshold:.3f}")
    print(f"{legalUser}: Average Precision {average_precision[legalUser]:.3f} ")   

num_processes = 8  # Adjust this based on your system capabilities

if __name__ == "__main__":
    # Set the number of processes you want to use
    num_processes = 8  # Adjust this based on your system capabilities
    manager = Manager()
    fpr = manager.dict()
    tpr = manager.dict()
    roc_auc = manager.dict()
    precission = manager.dict()
    recall = manager.dict()
    average_precision = manager.dict()
    # Create a ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Execute the process_user function for each legalUser in parallel
        # executor.map(process_user_svm, users, [fpr] * len(users), [tpr] * len(users), [roc_auc] * len(users), [precission] * len(users), [recall] * len(users), [average_precision] * len(users) )
        # executor.map(process_user_oc_svm, users, [fpr] * len(users), [tpr] * len(users), [roc_auc] * len(users), [precission] * len(users), [recall] * len(users), [average_precision] * len(users) )
        # executor.map(process_user, users, [fpr] * len(users), [tpr] * len(users), [roc_auc] * len(users), [precission] * len(users), [recall] * len(users), [average_precision] * len(users) )
        executor.map(process_user_iso_forest, users, [fpr] * len(users), [tpr] * len(users), [roc_auc] * len(users), [precission] * len(users), [recall] * len(users), [average_precision] * len(users) )

    # After the loop is finished, you can proceed with the rest of your code
    path = 'wykresy'
    roc_filename = f'IF_ROC_balabit_{extracted_number}_{balanced}'
    pr_filename = f'IF_PR_balabit_{extracted_number}_{balanced}'
    roc_path = os.path.join(path, roc_filename)
    pr_path = os.path.join(path, pr_filename)
    plotROCs(dict(fpr), dict(tpr), dict(roc_auc), users,False, True, roc_path)
    plot_precisions_recalls(dict(precission), dict(recall), dict(average_precision), True, pr_path)