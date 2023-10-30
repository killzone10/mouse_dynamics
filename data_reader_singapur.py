from data_reader import *
from data_processer_singapur import *
import pandas as pd

class DataReaderSingapur (DataReader):
    ## init creates dataprocesser ## 
   def __init__(self, dataset, users,  limit = 9999):
        super().__init__(dataset, users, False, limit)
        self.processor = DataProcesserSingapur(users, limit)
        self.stolenSessions = self.create_stolen_sessions(STOLEN_SESSIONS_FILEPATH)
   
   def create_stolen_sessions(self, file_path):
      try:
         df = pd.read_excel(file_path, na_filter=False)
         stolen_sessions = {}

         for idx, record in enumerate(df['Victim']):
            date = df['Attack period'][idx]
            stolen_sessions[record] = {'Tuesday': date}

         for idx, record in enumerate(df['Victim.1']):
            date = df['Attack period.1'][idx]
            stolen_sessions[record] = {'Thursday': date}

         return stolen_sessions
      except Exception as e:
         raise Exception(f"An error occurred while processing the Excel file: {str(e)}")



   ## DATA FROM 20-03-2017 to 24-03-2017 NEEDED
   def processDataWithoutLabels(self): ## TODO LIMT SHOULD BE CHANGED HERE SO DATA IS BALANCED !!
        self.supervised = False
        if self.supervised == True:
            raise ValueError("The boolean value cant be False in this situation")
        self.createUnsupervisedFilename()
        self.createFile()
        dirs = os.listdir(self.path[0])
        for user in dirs:
            if int(user) not in self.users:
                continue # TODO
            sessions = os.listdir(self.path[0] + '\\' + user)
            limit = int(self.limit/len(sessions)) ## TODO THINK ABOUT THAT  
            for session in sessions:
                path = os.path.join(self.path[0], user, session)
                print(path)
                self.processor.createProcessedCSV(path, user, self.fileName, limit) ### Tworzenie CSVileName, limit, self.supervised, legality = 1) ### Tworzenie CSV
        
          