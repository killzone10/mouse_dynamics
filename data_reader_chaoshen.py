from data_reader import *
from data_processer_chaoshen import *


class DataReaderChaoShen (DataReader): ## TODO ADD CHECK IF USERS EXIST IN THE BEGGINING
    def __init__(self, dataset, users, limit):
        super().__init__(dataset, users, False, limit)
        self.processor = DataProcesserChaoshen(users, limit)

 

    def processDataWithoutLabels(self): ## TODO LIMT SHOULD BE CHANGED HERE SO DATA IS BALANCED !!
        self.supervised = False
        if self.supervised == True:
            raise ValueError("The boolean value cant be False in this situation - use setSupervised() method first")
        
        self.createUnsupervisedFilename()
        if self.checkIfFileExist():
            print("File already exist")
            return
        
        self.createFile()
      
        dirs = os.listdir(self.path[0])
        for user in dirs:
            if int(user) not in self.users:
                continue # TODO
            sessions = os.listdir(self.path[0] + '\\' + user)
            limit = int(self.limit/len(sessions)) ## TODO THINK ABOUT THAT  
            for session in sessions:
                path = os.path.join(self.path[0], user, session)
                self.processor.createProcessedCSV(path, user, self.file, limit) ### Tworzenie CSV
                    
        self.closeFile()
