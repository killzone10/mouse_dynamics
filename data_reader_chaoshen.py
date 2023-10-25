from data_reader import *
from data_processer_chaoshen import *


class DataReaderChaoShen (DataReader):
     def __init__(self, name, breed):
         super().__init__(dataset, users, supervised, limit)
         self.processor = DataProcesserBalabit(users, limit)

 

    def processDataWithoutLabels(self): ## TODO LIMT SHOULD BE CHANGED HERE SO DATA IS BALANCED !!
        self.supervised = False
        if self.supervised == True:
            raise ValueError("The boolean value cant be False in this situation")
        self.createUnsupervisedFilename()

        self.createFile()
        dirs = os.listdir(self.path[0])
        for dir in dirs:
            user = dir.split('user')
            user = int(user[1])
            if user not in self.users:
                continue # TODO
            sessions = os.listdir(self.path[0] + '\\' + dir)
            limit = int(self.limit/len(sessions)) ## TODO THINK ABOUT THAT  
            for session in sessions:
                path = os.path.join(self.path[0], dir, session)
                self.processor.createProcessedCSV(path, user, self.fileName, limit, self.supervised, legality = 1) ### Tworzenie CSV
        
                    
   