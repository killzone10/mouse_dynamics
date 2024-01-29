from analyser import *


class legalityAnalyser(Analyser):
    def __init__(self, filename):
        super().__init__(filename)

    ## THIS PART IS BEING DONE IN ORDER TO CREATE LABELS FOR USERS FROM DATA WITHOUT LABELS ## 
    def selectNegativeSamples(self, numberOfSamples, balanced):
        otherUsersData = self.df['legality'] != 1
        if balanced == True:
            datasetNegatives = self.df[otherUsersData].drop(columns=['userid']).sample(numberOfSamples) ## DROPPING USERID is important ?
        else:
            datasetNegatives = self.df[otherUsersData].drop(columns=['userid'])
        array_negative = copy.deepcopy(datasetNegatives.values)
        return array_negative
    

    
    def getNumberOfSamples(self, array_positive):
        numberOfSamples = array_positive.shape[0]
        return numberOfSamples
    


    def selectPositiveSamples(self):
        # user_positive_data = self.df.loc[self.df.iloc[:, -1].isin([1])] 
        user_labels = self.df['legality'] == 1
        array_possitive = self.df[user_labels].drop(columns=['userid'])
        array_positive = copy.deepcopy(array_possitive.values)
        return array_positive
    

    def createTrainingData(self, balanced = True):
        array_positive = self.selectPositiveSamples()
        numberOfSamples = self.getNumberOfSamples(array_positive)
        array_negative = self.selectNegativeSamples(numberOfSamples, balanced)
        data_set = self.concatenateData(array_positive, array_negative)
        return data_set


    