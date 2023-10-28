from data_reader import *
from data_processer_chaoshen import *


class DataReaderChaoShen (DataReader):
    def __init__(self, dataset, users, supervised, limit):
        super().__init__(dataset, users, supervised, limit)
        self.processor = DataProcesserChaoshen(users, limit)

 

    def processDataWithoutLabels(self): ## TODO LIMT SHOULD BE CHANGED HERE SO DATA IS BALANCED !!
        self.supervised = False
        if self.supervised == True:
            raise ValueError("The boolean value cant be False in this situation")
        self.createUnsupervisedFilename()

        self.createFile()
        dirs = os.listdir(self.path[0])
        print(dirs)
        for user in dirs:
            if int(user) not in self.users:
                continue # TODO
            sessions = os.listdir(self.path[0] + '\\' + user)
            print(sessions)
            limit = int(self.limit/len(sessions)) ## TODO THINK ABOUT THAT  
            for session in sessions:
                path = os.path.join(self.path[0], user, session)
                self.processor.createProcessedCSV(path, user, self.fileName, limit) ### Tworzenie CSV
        
                    
   