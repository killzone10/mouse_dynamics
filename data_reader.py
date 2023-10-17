from utils.consts import *
from data_processer import DataProcesser
import os 

class DataReader:
    def __init__(self, dataset, users, test,  limit = 9999):
        self.users = users
        self.limit = limit # not neeeded right now
        self.fileName = f'users{users}_limit{limit}.csv'
        # TODO IF LEN add error
        if test == True:
            self.path = os.path.join(BASE_FOLDER[dataset], TEST_FOLDER[dataset])
            
        else:
            self.path = os.path.join(BASE_FOLDER[dataset], TRAINING_FOLDER[dataset])

        self.processor = DataProcesser(users, limit)

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
    
    def __createFile(self, legality):
        self.fileName = open(self.fileName, "w")
        if not legality:
            self.fileName.write(ACTION_CSV_HEADER)
        else:
            self.fileName.write(ACTION_CSV_HEADER_LEGALITY)

    def getFile(self):
        return self.fileName
    

    def processData(self, legality):
        self.__createFile(legality)
        dirs = os.listdir(self.path)
        for dir in dirs:
            user = dir.split('user')
            user = int(user[1])
            if user not in self.users:
                continue # TODO
            sessions = os.listdir(self.path + '\\' + dir)
            limit = int(self.limit/len(sessions)) ## TODO THINK ABOUT THAT 
            for session in sessions:
                path = os.path.join(self.path, dir, session)
                #TODO CHECK LEGALITY HERE !
                self.processor.createProcessedCSV(path, user, self.fileName, limit, legality) ### Tworzenie CSV

