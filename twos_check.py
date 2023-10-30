from data_reader_singapur import *

users = [1,2,3]
data_reader_singapur = DataReaderSingapur(SINGAPUR, users, 300)
data_reader_singapur.processDataWithoutLabels()