
## READ  CONSTS ##
from utils.consts import *
from  data_reader import DataReader
from data_reader_babalit import *
users = [7,9,15,16,20,21,23,29]
# users = [7,9,15,16,20,21,23,29,35]
# reader = DataReader(BALABIT, users, False, limit = 300) 
# reader.processData(legality = False)

# ## READ CHOSEN FILES ##


balabit_reader = DataReaderBalabit(BALABIT, users, True, limit = 300)
balabit_reader.processDataWithLabels(7, training = True, test = True, supervised = True)
balabit_reader.processDataWithoutLabels()



