from data_reader_chaoshen import *
from utils.consts import *


users = [i for i in range(1, 29)]
users = [1,2,4]
chaoshen_reader = DataReaderChaoShen(CHAOSHEN, users, False, 1000)
chaoshen_reader.processDataWithoutLabels()