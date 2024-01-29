from sklearn.preprocessing import MinMaxScaler
import numpy as np
from custom_dataset import *

class NeuralDataCreator():
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = dataset[:, :-1]
        self.y = dataset[:, -1]
        self.scaler = MinMaxScaler()

    def extract_positive_negative(self):
        x_negative = self.X[self.y == 0]
        x_positive = self.X[self.y == 1]
        y_pos = self.y[self.y == 1] 
        y_neg = self.y[self.y == 0] 
        return x_negative, x_positive, y_neg, y_pos

    def scale_data(self, x_negative, x_positive):
        x_negative = self.scaler.fit_transform(x_negative)
        x_positive = self.scaler.fit_transform(x_positive)
        return x_negative, x_positive

    def create_data(self, test_ratio, threshold_ratio):
        x_negative, x_positive, y_neg, y_pos = self.extract_positive_negative()
        x_negative, x_positive = self.scale_data(x_negative, x_positive)

        # Split positive samples into train and validation sets
        pos_size = len(x_positive)
        train_size = int((1 - test_ratio) * pos_size)

        x_training = x_positive[:train_size]
        y_training = y_pos[:train_size]

        x_validation = x_negative[:train_size]
        y_validation = y_neg[:train_size]

        # Threshold Data
        threshold_size = int(threshold_ratio * pos_size)
        threshold_index = train_size + threshold_size

        x_threshold_positive = x_positive[train_size:threshold_index]
        x_threshold_negative = x_negative[train_size:threshold_index]

        y_threshold_positive = y_pos[train_size:threshold_index]
        y_threshold_negative = y_neg[train_size:threshold_index]

        x_threshold = np.concatenate((x_threshold_positive, x_threshold_negative))
        y_threshold = np.concatenate(( y_threshold_positive, y_threshold_negative))

        # Test Data
        x_test_positive = x_positive[threshold_index:-1]
        x_test_negative = x_negative[threshold_index:-1]

        y_test_positive = y_pos[threshold_index:-1]
        y_test_negative = y_neg[threshold_index:-1]

        x_test = np.concatenate((x_test_positive, x_test_negative))
        y_test = np.concatenate((y_test_positive, y_test_negative))

        return x_training, y_training, x_validation, y_validation, x_threshold_positive, x_threshold_negative, y_threshold, x_test, y_test


    def create_datasets(self, x_training, x_validation, x_threshold_positive, x_threshold_negative , x_test):

        x_training_dataset = CustomDataset( x_training, transform = transforms.ToTensor())
        x_validation_dataset = CustomDataset( x_validation, transform = transforms.ToTensor())
        x_threshold_dataset_positive = CustomDataset( x_threshold_positive, transform = transforms.ToTensor())
        x_threshold_dataset_negative = CustomDataset( x_threshold_negative, transform = transforms.ToTensor())

        x_test_dataset = CustomDataset( x_test, transform = transforms.ToTensor())

        return x_training_dataset, x_validation_dataset,  x_threshold_dataset_positive, x_threshold_dataset_negative, x_test_dataset