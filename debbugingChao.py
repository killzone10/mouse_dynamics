from data_reader_chaoshen import *
from utils.consts import *


# users = [i for i in range(1, 29)]
# users = [1,2,4]
# chaoshen_reader = DataReaderChaoShen(CHAOSHEN, users, False, 1000)
# chaoshen_reader.processDataWithoutLabels()
users = [7,9,15,16,20,21,23,29, 35]

from data_reader_babalit import *
balabit_reader = DataReaderBalabit(BALABIT, users, False, limit = 1000)
balabit_reader.processDataWithoutLabels()
