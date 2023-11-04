from utils.consts import *
from data_processer import DataProcesser
import os 

## initialiation of dataset, limit, users and path ##
## dataset is the name of the dataset - its used to create a self.name
# users are users analyzed
#supervised or not supervised learning 
# limit = how many actions should be analyzed #

class DataReader:  
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

        self.supervised = supervised
        self.resetFileName()

        if not os.path.exists(OUTPUT_FILE):
            os.makedirs(OUTPUT_FILE)
        ## if supervised add the path of test files in balabit ##
        if supervised == True: ## TODO add balabit requiretn
            self.path.append(os.path.join(BASE_FOLDER[dataset], TEST_FOLDER[dataset]))
            
    ## getters ##
    def readUsers(self):
        return self.user

    def setSupervised(self, supervised):
        if supervised == True:
            self.supervised = supervised
            if len(self.path) == 1:
                self.path.append(os.path.join(BASE_FOLDER[self.dataset], TEST_FOLDER[self.dataset]))

        if supervised == False:
            self.supervised = supervised
            if len(self.path) == 2:
                self.path.pop()

    ## CREATE FILE (its important) ##
    def createFile(self):
        try:
            self.file = open(self.fileName, "w")
            if not self.supervised:
                self.file.write(ACTION_CSV_HEADER[self.dataset])
            else:
                self.file.write(ACTION_CSV_HEADER_LEGALITY[self.dataset])
        except Exception as e:
            raise Exception(f"An error occurred while processing the opening the  file: {str(e)}")
        
    ## get file ##
    def getFileName(self):
        return self.fileName
    
    ## reset path ## (used when changing the function of creating the dataset)
    def resetFileName(self):
        fileName = f'{self.name}_dataset_users{self.users}_limit{self.limit}_labels{self.supervised}.csv'    
        fileName = os.path.join(OUTPUT_FILE, fileName)
        self.fileName = fileName

    ## path ##
    def createFileNamePath(self, training, test, legalUser):
        self.resetFileName()
        if training == True:
            base, extension = os.path.splitext(self.fileName)
            self.fileName = f'{base}_legalUser{legalUser}_Training{extension}'

        if test == True:
            base, extension = os.path.splitext(self.fileName)
            self.fileName = f'{base}_Test{extension}'

    ## path ##
    def createUnsupervisedFilename(self):
        self.fileName = os.path.join(OUTPUT_FILE,f'{self.name}_dataset_users{self.users}_limit{self.limit}_labels{self.supervised}.csv')

        
    def checkIfFileExist(self):
        if os.path.exists(self.fileName):
            return True
        else:
            return False

    def closeFile(self):
        self.file.close()
