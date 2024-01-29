from analyser import *



class nonLegalityAnalyser(Analyser):
    def __init__(self, filename):
        super().__init__(filename)

    ## THIS PART IS BEING DONE IN ORDER TO CREATE LABELS FOR USERS FROM DATA WITHOUT LABELS ## 
    def selectNegativeSamplesWithLabel(self, legalUser, numberOfSamples, balanced):
        otherUsersData = self.df['userid'] != legalUser
        if balanced == True:      
            datasetNegatives = self.df[otherUsersData].sample(numberOfSamples)
        else:
            datasetNegatives = self.df[otherUsersData]
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
    

    def createTrainingDataWithLabel(self, legalUser, balanced = True):
        array_positive = self.selectPositiveSamplesWithLabel(legalUser)
        numberOfSamples = self.getNumberOfSamples(array_positive)
        array_negative = self.selectNegativeSamplesWithLabel(legalUser, numberOfSamples, balanced)
        data_set = self.concatenateData(array_positive, array_negative)
        return data_set

    ## END OF THE LABEL PART ##


    def selectPositiveSamples(self, legalUser):
        user_positive_data = self.df.loc[self.df.iloc[:, -1].isin([legalUser])] 
       # numberOfSamples = user_positive_data.shape[0]
        array_positive = copy.deepcopy(user_positive_data.values)
    
    def selectNegativeSamples(self, legalUser, numberOfSamples):
        otherUsersData = self.df['userid'] != legalUser
        datasetNegatives = self.df[otherUsersData].sample(numberOfSamples)
        array_negative = copy.deepcopy(datasetNegatives.values)
        return array_negative
    

    def createTrainingData(self, legalUser):
        array_positive = self.selectPositiveSamples(legalUser)
        numberOfSamples = self.getNumberOfSamples(array_positive)
        array_negative = self.selectNegativeSamples(legalUser, numberOfSamples)
        data_set = self.concatenateData(array_positive, array_negative)
        return data_set


    def createDataForUnsupervised(self, legalUser, TEST_SIZE, balanced = True):
        user_positive_data = self.df.loc[self.df.iloc[:, -1].isin([legalUser])] 
       # numberOfSamples = user_positive_data.shape[0]
        array_positive = copy.deepcopy(user_positive_data.values)
        array_positive[:, -1] = 1 # ADD LABEL 1 to the  true user

        num_positive_samples = int(TEST_SIZE * len(array_positive))
        ## NEGATIVE DATA
        test_data_negative = self.selectNegativeSamplesWithLabel(legalUser, num_positive_samples, balanced)
        ## POSITIVE DATA
        test_data_positive = array_positive[-num_positive_samples:]
        # READJUST OF MAIN DATA ## 
        array_positive = array_positive[:-num_positive_samples]
        training_data = pd.DataFrame(array_positive).values

        test_data = self.concatenateData(test_data_negative, test_data_positive)
        # test_data = np.concatenate(test_data_positive, test_data_negative)
        return  training_data, test_data
        