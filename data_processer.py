from utils.consts import *
import csv
import os
from utils.helper_functions import *

class DataProcesser:
    def __init__(self, user, limit):
        self.user = user
        self.limit = limit
        self.fileName = f'users{user}.csv'

    
    def __computeFeatures(self, x, y, t, action, file, start, stop, user):
        pass


    def __queueAction(self, x, y, t, actionCode, action_file, n_from, n_to, user):
        pass
    #one MM action
    def __processMM(self, x, y, t, action_file, start, stop, user):
        pass

    # one DD action
    def __processDD(self, x, y, t,action_file, start, stop, user):
        pass

    # one SS action
    def __processSS(self, x, y, t, action_file, start, stop, user): # to jest do dodania
        pass
   
    def __processPC(self, x, y, t, action_file, start, stop, user): # to jest do dodania
       pass
   
    def __processCombinedPC(self, actions, action_file, start, stop, user):
        pass
   
    def createProcessedCSV(self, path , user, fileName, limit, legality): ## check limit 
        pass