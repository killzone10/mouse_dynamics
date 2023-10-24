from analyser import *
import copy



class nonLegalityAnalyser(Analyser):
    def __init__(self, filename):
        super().__init__(filename)

    ## THIS PART IS BEING DONE IN ORDER TO CREATE LABELS FOR USERS FROM DATA WITHOUT LABELS ## 
    def selectNegativeSamplesWithLabel(self, legalUser, numberOfSamples):
        otherUsersData = self.df['userid'] != legalUser
        datasetNegatives = self.df[otherUsersData].sample(numberOfSamples)
        array_negative = copy.deepcopy(datasetNegatives.values)
        array_negative[:, -1] = 0
        return array_negative
    
    def selectPositiveSamplesWithLabel(self, legalUser):
        user_positive_data = self.df.loc[self.df.iloc[:, -1].isin([legalUser])] 
       # numberOfSamples = user_positive_data.shape[0]
        array_positive = copy.deepcopy(user_positive_data.values)
        array_positive[:, -1] = 1 # ADD LABEL 1 to the  true user
        return  array_positive
    
    def getNumberOfSamples(self, array_positive):
        numberOfSamples = array_positive.shape[0]
        return numberOfSamples
    

    def createTrainingDataWithLabel(self, legalUser):
        array_positive = self.selectPositiveSamplesWithLabel(legalUser)
        numberOfSamples = self.getNumberOfSamples(array_positive)
        array_negative = self.selectNegativeSamplesWithLabel(legalUser, numberOfSamples)
        data_set = self.concatenateData(array_positive, array_negative)
        return data_set

    ## END OF THE LABEL PART 

    
