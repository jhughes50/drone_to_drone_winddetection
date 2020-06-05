import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.externals import joblib
from utils.classifications_utils import *
from utils.data_processing_utils_test import *
#from utils.data_visualization_utils import *
from utils.metrics_utils import *
from utils.grid_search_utils import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from transformers.featureGenerator import FeatureGenerator
from sklearn.ensemble import GradientBoostingClassifier


gyro_scores = []
acc_scores  = []


print('TESTING...')

for f in range(15):
    
   # Reset all values, so you can re-play the notebook without kernal restart
   no_wind_data_gyro = 0
   level_1_wind_gyro = 0
   no_wind_data_acc = 0
   level_1_wind_acc = 0

   # Load data
   no_wind_data = load_data(0, 35, 'project3', 'drone1')
   level_1_wind = load_data(1, 35, 'project3', 'drone1')

   # Drops unnecessary data points 
   no_wind_data = no_wind_data.drop(columns=['label', 'stabilizer.pitch', 'stabilizer.roll', 'stabilizer.yaw', 'stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z', 'timestamp_end', 'timestamp_start'])
   level_1_wind = level_1_wind.drop(columns=['label', 'stabilizer.pitch', 'stabilizer.roll', 'stabilizer.yaw', 'stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z', 'timestamp_end', 'timestamp_start'])

   # Seperates the data into 4 different categories
   no_wind_data_gyro = no_wind_data.drop(columns=['acc.x', 'acc.y', 'acc.z'])

   level_1_wind_gyro = level_1_wind.drop(columns=['acc.x', 'acc.y', 'acc.z'])

   no_wind_data_acc = no_wind_data.drop(columns=['gyro.x', 'gyro.y', 'gyro.z'])

   level_1_wind_acc = level_1_wind.drop(columns=['gyro.x', 'gyro.y', 'gyro.z'])   

   # Convert from dict to panda dataframe
   no_wind_data_gyro = {"gyro": no_wind_data_gyro}
   level_1_wind_gyro = {"gyro": level_1_wind_gyro}

   no_wind_data_acc = {"acc": no_wind_data_acc}
   level_1_wind_acc = {"acc": level_1_wind_acc}

   label_0 = [0 for x in range(len(no_wind_data_gyro['gyro']))]
   label_1 = [1 for x in range(len(level_1_wind_gyro['gyro']))]

   X_train_acc_0, X_test_acc_0, y_train_acc_0, y_test_acc_0 = \
   train_test_split(no_wind_data_acc['acc'], label_0, test_size=0.20, shuffle=True)

   X_train_gyro_0, X_test_gyro_0, y_train_gyro_0, y_test_gyro_0 = \
   train_test_split(no_wind_data_gyro['gyro'], label_0, test_size=0.20, shuffle=True)

   X_train_acc_1, X_test_acc_1, y_train_acc_1, y_test_acc_1 = \
   train_test_split(level_1_wind_acc['acc'], label_1, test_size=0.20, shuffle=True)

   X_train_gyro_1, X_test_gyro_1, y_train_gyro_1, y_test_gyro_1 = \
   train_test_split(level_1_wind_gyro['gyro'], label_1, test_size=0.20, shuffle=True)

   X_train_gyro = X_train_gyro_0.append(X_train_gyro_1)
   y_train_gyro = np.hstack((y_train_gyro_0, y_train_gyro_1))

   X_test_gyro = X_test_gyro_0.append(X_test_gyro_1)
   y_test_gyro = np.hstack((y_test_gyro_0, y_test_gyro_1))

   X_train_acc = X_train_acc_0.append(X_train_acc_1)
   y_train_acc = np.hstack((y_train_acc_0, y_train_acc_1))

   X_test_acc = X_test_acc_0.append(X_test_acc_1)
   y_test_acc = np.hstack((y_test_acc_0, y_test_acc_1))

   feature_generator_gyro = FeatureGenerator(1, 'gyro')
   feature_generator_gyro.fit(X_train_gyro, 2)

   X_train_gyro = feature_generator_gyro.transform(X_train_gyro)
   y_train_gyro = adjust_label_amount(y_train_gyro, 2)

   feature_generator_gyro_test = FeatureGenerator(1, 'gyro')
   feature_generator_gyro_test.fit(X_test_gyro, 2)
   
   X_test_gyro = feature_generator_gyro_test.transform(X_test_gyro)
   y_test_gyro = adjust_label_amount(y_test_gyro, 2)

   feature_generator_acc = FeatureGenerator(1, 'acc')
   feature_generator_acc.fit(X_train_acc, 2)

   X_train_acc = feature_generator_acc.transform(X_train_acc)
   y_train_acc = adjust_label_amount(y_train_acc, 2)
   
   feature_generator_acc_test = FeatureGenerator(1, 'acc')
   feature_generator_acc_test.fit(X_test_acc, 2)

   X_test_acc = feature_generator_acc_test.transform(X_test_acc)
   y_test_acc = adjust_label_amount(y_test_acc, 2)

   clf_gyro = GradientBoostingClassifier()
   clf_gyro.fit(X_train_gyro, y_train_gyro)

   gyro_scores.append(clf_gyro.score(X_test_gyro, y_test_gyro))

   clf_acc = GradientBoostingClassifier()
   clf_acc.fit(X_train_acc, y_train_acc)   

   acc_scores.append(clf_acc.score(X_test_acc, y_test_acc))
       
gyro_average = sum(gyro_scores)/ len(gyro_scores)
acc_average  = sum(acc_scores)/ len(acc_scores)

print('Gyro Average: ', gyro_average)
print('Acc Average: ', acc_average)

array_gyro = np.asarray(gyro_scores)
array_acc  = np.asarray(acc_scores)

np.savetxt('gyro.txt', array_gyro)
np.savetxt('acc.txt',array_acc)





    

