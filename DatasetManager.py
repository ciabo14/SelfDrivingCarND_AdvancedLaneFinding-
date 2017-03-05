import os
import numpy as np
from sklearn.model_selection import train_test_split

test_p = .2

dataset_positive = "./Dataset/vehiclesComplete"
dataset_negative = "./Dataset/non-vehicles"

class DatasetManager():

    def __init__(self):
        pass

    # generate the train/test split starting from the availabel images. All the images are included in a different folder depending on the sequence
    
    def compute_dataset(self):
        
        train_set = []
        train_set_labels = []
        test_set = []
        test_set_labels = []
        files1 = []
        labels1 = []
        files2 = []
        labels2 = []
        for (path, dirs, f) in os.walk(dataset_positive):
            np.random.shuffle(f)
            files1 = ["{}/{}".format(dataset_positive,file_name) for file_name in f]
            labels1 = [1 for i in files1]
        for (path, dirs, f) in os.walk(dataset_negative):
            np.random.shuffle(f)
            files2 = ["{}/{}".format(dataset_negative,file_name) for file_name in f]
            labels2 = [0 for i in files2]
        
        X_train, X_test, y_train, y_test = train_test_split(files1+files2, labels1+labels2, test_size=test_p, random_state=42)
        train_set.extend(X_train)
        test_set.extend(X_test)
        train_set_labels.extend(y_train)
        test_set_labels.extend(y_test)
       
        return train_set, train_set_labels, test_set, test_set_labels