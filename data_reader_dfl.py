from data_reader import *
from data_processer_dfl import *


class DataReaderDfl (DataReader):
    ## init creates dataprocesser ## 
    def __init__(self, dataset, users, supervised,  limit = 9999):
        super().__init__(dataset, users, supervised, limit)
        self.processor = DataProcesserDfl(users, limit)


    def processDataWithoutLabels(self): ## TODO LIMT SHOULD BE CHANGED HERE SO DATA IS BALANCED !!
        self.supervised = False
        if self.supervised == True:
            raise ValueError("The boolean value cant be False in this situation")
        self.createUnsupervisedFilename()
        if self.checkIfFileExist():
            print("File already exist")
            return
        self.createFile()
       

        dirs = os.listdir(self.path[0])
        for dir in dirs:
            user = dir.split('User')
            user = int(user[1])
            if user not in self.users:
                continue # TODO
            sessions = os.listdir(self.path[0] + '\\' + dir)
            limit = int(self.limit/len(sessions)) ## TODO THINK ABOUT THAT  
            for session in sessions:
                path = os.path.join(self.path[0], dir, session)
                self.processor.createProcessedCSV(path, user, self.file, limit, self.supervised, legality = 1) ### Tworzenie CSV
        

        self.closeFile()
