from data_reader import *
from data_processer_singapur import *
import pandas as pd
from datetime import datetime


class DataReaderSingapur (DataReader):
    ## init creates dataprocesser ## 
   def __init__(self, dataset, users,  limit = 100000):
        super().__init__(dataset, users, False, limit)
        self.processor = DataProcesserSingapur(users, limit)
        self.stolenSessions = self.create_stolen_sessions(STOLEN_SESSIONS_FILEPATH)
   
   def create_stolen_sessions(self, file_path):
      tuesday = datetime(2017, 3, 21)
      thursday = datetime(2017,3, 23)
      try:
         df = pd.read_excel(file_path, na_filter = False)
         stolen_sessions = {}
         for idx, record in enumerate(df['Victim']):
            if record == "":
               continue
            date = df['Attack period'][idx]
            splitted = date.split("-")
            time_beg = splitted[0].replace('.',':')
            time_end = splitted[1].replace('.',':')
            beggining = datetime.strptime(f"{tuesday:%Y-%m-%d} {time_beg}", "%Y-%m-%d %H:%M ")            
            end = datetime.strptime(f"{tuesday:%Y-%m-%d} {time_end}", "%Y-%m-%d %H:%M")            
            stolen_sessions[record] =  [beggining, end]

         for idx, record in enumerate(df['Victim.1']):
            if record == "":
                continue
            date = df['Attack period.1'][idx]
            splitted = date.split("-")
            time_beg = splitted[0].replace('.',':')
            time_end = splitted[1].replace('.',':')
            beggining = datetime.strptime(f"{thursday:%Y-%m-%d} {time_beg}", "%Y-%m-%d %H:%M ")            
            end = datetime.strptime(f"{thursday:%Y-%m-%d} {time_end}", "%Y-%m-%d %H:%M")            

        # stolen_sessions[record] = {tuesday: date}
            stolen_sessions[record] =  [beggining, end]
         return stolen_sessions
      except Exception as e:
         raise Exception(f"An error occurred while processing the Excel file: {str(e)}")


   ## DATA FROM 20-03-2017 to 24-03-2017 NEEDED
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
      for user in dirs:
         if int(user) not in self.users:
            continue # TODO

         sessions = os.listdir(self.path[0] + '\\' + user)
         limit = int(self.limit/len(sessions)) ## TODO THINK ABOUT THAT  

         for session in sessions:
            path = os.path.join(self.path[0], user, session)
            self.processor.createProcessedCSV(path, user, self.file, limit, self.stolenSessions) ### Tworzenie CSVileName, limit, self.supervised, legality = 1) ### Tworzenie CSV

      self.closeFile()
      
  


   def processTestData(self):
      self.supervised = False
      if self.supervised == True:
         raise ValueError("The boolean value cant be False in this situation")
      dirs = os.listdir(self.path[0])
      for user in dirs:
         if int(user) not in self.users:
            continue # TODO
        
         userName = f"User{user}"
         if userName in self.stolenSessions:  
            fileName = os.path.join(TEST_FILES, user)
            fileName = open(fileName, "w")
            sessions = os.listdir(self.path[0] + '\\' + user)
            limit = 10000000 ## TODO THINK ABOUT THAT  
            for session in sessions:
                  path = os.path.join(self.path[0], user, session)
                  self.processor.createProcessedCSV(path, user, fileName, limit, self.stolenSessions) 
         