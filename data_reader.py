from utils.consts import *
from data_processer import DataProcesser
import os 

class DataReader: ## TODO THINK ABOUT POLYMOPHYSM 
    ## initialiation of dataset, limit, users and path ##
    def __init__(self, dataset, users, supervised, limit = 9999):
        self.users = users
        self.limit = limit # not neeeded right now
        self.dataset = dataset
        self.path = [os.path.join(BASE_FOLDER[dataset], TRAINING_FOLDER[dataset])]
        if dataset == 0:
            self.name = 'balabit'
        elif dataset == 1:
            self.name = 'chaoshen'
        elif dataset == 2:
            self.name = "Singapur"
        elif dataset == 3:
            self.name = "dfl"
            
        self.fileName = f'{self.name}_dataset_users{users}_limit{limit}_labels{supervised}.csv'
        ## if supervised add the path of test files in balabit ##
        if supervised == True: ## TODO add balabit requiretn
            self.path.append(os.path.join(BASE_FOLDER[dataset], TEST_FOLDER[dataset]))
        self.supervised = supervised

    ## getters ##
    def readUsers(self):
        return self.user


    def getUser(self, index):
        if 0 <= index < len(self.user):
            return self.user[index]
        else:
            return None
    ## Setters ## 
    def setUser(self, index, value):
        if 0 <= index < len(self.user):
            self.user[index] = value
        else:
            print("Index out of range.")

    ## CREATE FILE (its important) ##
    def createFile(self):
        try:
            self.fileName = open(self.fileName, "w")
            if not self.supervised:
                self.fileName.write(ACTION_CSV_HEADER[self.dataset])
            else:
                self.fileName.write(ACTION_CSV_HEADER_LEGALITY[self.dataset])
        except Exception as e:
            raise Exception(f"An error occurred while processing the opening the  file: {str(e)}")
        
    ## get file ##
    def getFile(self):
        return self.fileName.name
    
    ## reset path ## (used when changing the function of creating the dataset)
    def resetFileName(self):
        self.fileName = f'{self.name}_dataset_users{self.users}_limit{self.limit}_labels{self.supervised}.csv'    

    ## path ##
    def createFileNamePath(self, training, test):
        self.resetFileName()
        if training == True:
            base, extension = os.path.splitext(self.fileName)
            self.fileName = f'{base}_Training{extension}'

        if test == True:
            base, extension = os.path.splitext(self.fileName)
            self.fileName = f'{base}_Test{extension}'

    ## path ##
    def createUnsupervisedFilename(self):
        self.fileName = f'{self.name}_dataset_users{self.users}_limit{self.limit}_labels{self.supervised}.csv'

        