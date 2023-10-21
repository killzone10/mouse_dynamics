from utils.consts import *
from data_processer import DataProcesser
import os 

class DataReader: ## TODO THINK ABOUT POLYMOPHYSM 
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
            self.name = "Sigapur"
        self.fileName = f'{self.name}_dataset_users{users}_limit{limit}_labels{supervised}.csv'
        if supervised == True:
            self.path.append(os.path.join(BASE_FOLDER[dataset], TEST_FOLDER[dataset]))
        
        self.supervised = supervised
    def readUsers(self):
        return self.user

    def getUser(self, index):
        if 0 <= index < len(self.user):
            return self.user[index]
        else:
            return None

    def setUser(self, index, value):
        if 0 <= index < len(self.user):
            self.user[index] = value
        else:
            print("Index out of range.")

    def appendUser(self, value):
        self.user.append(value)

    def userLength(self):
        return len(self.user)
    
    def createFile(self):
        self.fileName = open(self.fileName, "w")
        if not self.supervised:
            self.fileName.write(ACTION_CSV_HEADER[self.dataset])
        else:
            self.fileName.write(ACTION_CSV_HEADER_LEGALITY[self.dataset])

    def getFile(self):
        return self.fileName
    
    def readUsers(self):
        return self.user

    def getUser(self, index):
        if 0 <= index < len(self.user):
            return self.user[index]
        else:
            return None

    def setUser(self, index, value):
        if 0 <= index < len(self.user):
            self.user[index] = value
        else:
            print("Index out of range.")

    def appendUser(self, value):
        self.user.append(value)

    def userLength(self):
        return len(self.user)
    
    

    def processDataWithoutLabels(self):
        # self.__createFile(legality)
        # dirs = os.listdir(self.path)
        # for dir in dirs:
        #     user = dir.split('user')
        #     user = int(user[1])
        #     if user not in self.users:
        #         continue # TODO
        #     sessions = os.listdir(self.path + '\\' + dir)
        #     limit = int(self.limit/len(sessions)) ## TODO THINK ABOUT THAT 
        #     for session in sessions:
        #         path = os.path.join(self.path, dir, session)
        #         #TODO CHECK LEGALITY HERE !
        #         self.processor.createProcessedCSV(path, user, self.fileName, limit, legality) ### Tworzenie CSV
        pass

    def createFileNamePath(self, training, test):
        if training == True:
            base, extension = os.path.splitext(self.fileName)
            self.fileName = f'{base}_Training{extension}'

        if test == True:
            base, extension = os.path.splitext(self.fileName)
            self.fileName = f'{base}_Test{extension}'

    def createUnsupervisedFilename(self):
        self.fileName = f'{self.name}_dataset_users{self.users}_limit{self.limit}_labels{self.supervised}.csv'
