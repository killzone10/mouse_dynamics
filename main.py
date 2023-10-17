
## READ  CONSTS ##
from utils.consts import *
from  data_reader import DataReader
users = [7,9,15,16,20,21,23,29,35]
reader = DataReader(BALABIT, users, False, limit = 300) 
reader.processData(legality = False)

## READ CHOSEN FILES ##

