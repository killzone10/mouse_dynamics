from data_reader import *
from data_processer_balabit import *


class DataReaderBalabit (DataReader):
    ## init creates dataprocesser ## 
    def __init__(self, dataset, users, supervised,  limit = 9999):
        super().__init__(dataset, users, supervised, limit)
        self.processor = DataProcesserBalabit(users, limit)

    ## This function checks the csv file in order to get legality variable from test files ## 
    def check_legality(self):
        legality={}
        input_file  = open('Mouse-Dynamics-Challenge-master-2\public_labels.csv', "r")
        reader = csv.DictReader(input_file)
        for row in reader:
            fname = row['filename']
            is_illegal = row['is_illegal']
            # sessionid = str(fname[8:len(fname)]) ## same id sesji
            sessionid = str(fname)
            legality[sessionid] = 1-int(is_illegal) ##  zamiana is illegal na legal, zeby bylo czytelniejsze
        input_file.close()
        return legality

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
        
                    
    def processDataWithLabels(self, legalUser, training = True, test = False):
        self.supervised = True
        if self.supervised == False:
            raise ValueError("The boolean value cant be False in this situation")
     
        self.createFileNamePath(training, test)
        self.createFile()

        if legalUser not in self.users:
            raise ValueError("LegalUser doesnt exist")
                
        ## TRAINING FILES ## 
        if training:
            dirs = os.listdir(self.path[0])
            for dir in dirs:
                user = dir.split('user')
                user = int(user[1])
                if user not in self.users:
                    continue # TODO
               
                if user == legalUser:
                    legality = 1 
                else:
                    legality = 0
                sessions = os.listdir(self.path[0] + '\\' + dir)
                limit = int(self.limit/len(sessions)) ## TODO THINK ABOUT THAT 
                for session in sessions:
                    path = os.path.join(self.path[0], dir, session)
                    #TODO CHECK LEGALITY HERE !
                    self.processor.createProcessedCSV(path, user, self.fileName, limit, self.supervised, legality) ### Tworzenie CSV

        if test:
            legalityList = self.check_legality()
            dirs = os.listdir(self.path[1])
            for dir in dirs:
                user = dir.split('user')
                user = int(user[1])
                if user not in self.users:
                    continue # TODO
                sessions = os.listdir(self.path[1] + '\\' + dir)
                limit = int(self.limit/len(sessions)) ## TODO THINK ABOUT THAT 
                for session in sessions:
                    path = os.path.join(self.path[1], dir, session)
                    try:
                        if user == legalUser:
                            legality = legalityList[session] ## TODO IT CAN CREATE UNBALANCED DATASET
                        else:
                            legality = 0
                
                        self.processor.createProcessedCSV(path, user, self.fileName, limit, self.supervised, legality) ### Tworzenie CSV

                    except IndexError as e:
                        print(f"Index error {e}")
                    except Exception as e:
                        print(f"Unexcepted error {e}")
