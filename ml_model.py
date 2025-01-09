import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score

def load_data():
  # Download the data from Kaggle (replace with your downloaded file path)
  data_path = "C:\\Users\\admin\\Documents\\Mine Documents\\Data Sciene\\Mobile Price Range Predictor Web App\\train.csv"  
  data = pd.read_csv(data_path)
  return data


def preprocess_data(data):
    data = data[data['px_height']!=0]
    mean_sc_w = np.mean(data[data['sc_w'] != 0]['sc_w'])
    data.loc[data['sc_w'] == 0, 'sc_w'] = mean_sc_w
    pre_data = data.drop(columns=['fc','m_dep','mobile_wt','talk_time','three_g','blue','dual_sim','four_g','n_cores','pc','sc_h',
                                         'touch_screen','wifi','clock_speed','fc','m_dep','mobile_wt','talk_time','three_g'])
    return pre_data


def train_model(pre_data):
    X = pre_data[['battery_power', 'int_memory', 'px_height', 'px_width', 'ram', 'sc_w']]
    y = pre_data['price_range'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)
    lr = SVC()
    lr.fit(X_train, y_train)
    return lr


def evaluation(pre_data):
    X = pre_data[['battery_power', 'int_memory', 'px_height', 'px_width', 'ram', 'sc_w']]
    y = pre_data['price_range'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores_lr = cross_val_score(SVC(), X_train, y_train, cv=cv)
    cv_mean_score_lr = np.mean(scores_lr)*100
    lr = SVC()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accu= accuracy_score(y_test,y_pred)
    
    return accu, cv_mean_score_lr
    
    
def predict__(pre_data, lr):
    prediction = lr.predict(pre_data)
    return prediction
    