from data_reader import *



class DataReaderChaoShen (DataReader):
     def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed
