
## READ  CONSTS ##
from utils.consts import *
from  data_reader import DataReader
from data_reader_babalit import *
# users = [7,9,15,16,20,21,23,29,35]
# reader = DataReader(BALABIT, users, False, limit = 300) 
# reader.processData(legality = False)

# ## READ CHOSEN FILES ##
from utils.consts import *
from utils. plotting import *
from data_reader_babalit import *
from nonlegality_analyser import *




# users = [7,9,15,16,20,21,23,29, 35]
users = [7,9,15,16,20]

balabit_reader = DataReaderBalabit(BALABIT, users, True, limit = 300)
balabit_reader.processDataWithoutLabels()



path = balabit_reader.getFile()
balabitAnalyser = nonLegalityAnalyser(path)
print(balabitAnalyser.countActions())

from RFmodel import *

shuffle = True
fpr = {}
tpr = {}
roc_auc = {}

for legalUser in users:
    dataset = balabitAnalyser.createTrainingDataWithLabel(legalUser)
    X = dataset[:, 0:-1]
    y = dataset[:, -1]
    X_train, X_validation, y_train, y_validation = balabitAnalyser.trainingTestSplit(X, y, TEST_SIZE, shuffle)
    model = RFModel(dataset, users)
    fpr[legalUser], tpr[legalUser], thr = model.evaluate(X_train, y_train, X_validation, y_validation, legalUser)
    threshold = -1
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr[legalUser], tpr[legalUser])(x), 0., 1.) ## brentq znajdowanie 0 , interpid interpolacja
        threshold = interp1d(fpr[legalUser], thr)(eer)

    except (ZeroDivisionError, ValueError):
        print("Division by zero")

    roc_auc[legalUser] = auc(fpr[legalUser], tpr[legalUser])
    print(str(legalUser) + ": " + str(roc_auc[legalUser])+" threshold: "+str(threshold))

plotROCs(fpr, tpr, roc_auc, users)


