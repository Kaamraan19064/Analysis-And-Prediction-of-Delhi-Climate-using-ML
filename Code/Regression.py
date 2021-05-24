import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.dates as mdates
import warnings
import itertools
import dateutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV as gsc
from sklearn.linear_model import Ridge,Lasso
from sklearn.ensemble  import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def main ():


    # Using svm
    data=pd.read_csv('Original_with_dummies.csv')
    y = data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI'] = y
    S1,S2=AQI_SVM(data)
    S3,S4=AQI_Feature_importance_SVM(data)
    S5,S6=AQI_Domain_Knowledge_SVM(data)
    S7,S8=AQI_without_Domain_Knowledge_SVM(data)

##Linear Regression
    data=pd.read_csv('Original_with_dummies.csv')
    y = data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI'] = y

    LR1,LR2=AQI(data)
    LR3,LR4=AQI_Feature_importance(data)
    LR5,LR6==AQI_Domain_Knowledge(data)
    LR7,LR8=AQI_without_Domain_Knowledge(data)

## Predincting for next day
    data=pd.read_csv('Original_with_dummies.csv')
    y = data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI'] = y
    normalize(data)
    y=pd.read_csv('AQI_prediction_add.csv')
    LR_F1,LR_F2=AQI_Future(data,y.AQI_predicted)
    LR_F3,LR_F4=AQI_Feature_importance_Future(data,y.AQI_predicted)
    LR_F5,LR_F6=AQI_Domain_Knowledge_Future(data,y.AQI_predicted)
    LR_F7,LR_F8=AQI_without_Domain_Knowledge_Future(data,y.AQI_predicted)

##Predicting for Autumn Season
    data=pd.read_csv('autumn_data.csv')
    y = data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI'] = y
    data=pd.get_dummies(data, columns=[' _conds'], prefix = [' _conds'])
    data=pd.get_dummies(data, columns=[' _wdire'], prefix = [' _wdire'])
    data=pd.get_dummies(data, columns=['Type'], prefix = ['Type'])
    LR_A1,LR_A2=AQI(data)
    LR_A3,LR_A4=AQI_Feature_importance(data)
    LR_A5,LR_A6=AQI_Domain_Knowledge(data)
    LR_A7,LR_A8=AQI_without_Domain_Knowledge(data)

##Predicting for Summer Season
    data=pd.read_csv('summer_data.csv')
    y = data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI'] = y
    data=pd.get_dummies(data, columns=[' _conds'], prefix = [' _conds'])
    data=pd.get_dummies(data, columns=[' _wdire'], prefix = [' _wdire'])
    data=pd.get_dummies(data, columns=['Type'], prefix = ['Type'])
    LR_S1,LR_S2=AQI(data)
    LR_S3,LR_S4=AQI_Feature_importance(data)
    LR_S5,LR_S6=AQI_Domain_Knowledge(data)
    LR_S7,LR_S8=AQI_without_Domain_Knowledge(data)

##Predicting for Winter Season
    data=pd.read_csv('winter_data.csv')
    y = data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI'] = y
    data=pd.get_dummies(data, columns=[' _conds'], prefix = [' _conds'])
    data=pd.get_dummies(data, columns=[' _wdire'], prefix = [' _wdire'])
    data=pd.get_dummies(data, columns=['Type'], prefix = ['Type'])
    LR_W1,LR_W2=AQI(data)
    LR_W3,LR_W4=AQI_Feature_importance(data)
    LR_W5,LR_W6=AQI_Domain_Knowledge(data)
    LR_W7,LR_W8=AQI_without_Domain_Knowledge(data)


##Using Ridge
    data = pd.read_csv('Original_with_dummies.csv')
    y = data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI'] = y
    h = BestParams(data)
    ## Using all features
    R1,R2=AQI_Ridge(data,h)
    R3,R4=AQI_Feature_importance_Ridge(data,h)
    R5,R6=AQI_Domain_Knowledge_Ridge(data,h)
    R7,R8=AQI_without_Domain_Knowledge_Ridge(data,h)

    ##Future
    data = pd.read_csv('Original_with_dummies.csv')
    y = data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI'] = y
    h = BestParams(data)
    y = pd.read_csv('AQI_prediction_add.csv')
    R_F1,R_F2=AQI_Future_Ridge(data, y.AQI_predicted,h)
    R_F3,R_F4=AQI_Feature_importance_Future_Ridge(data, y.AQI_predicted,h)
    R_F5,R_F6=AQI_Domain_Knowledge_Future_Ridge(data, y.AQI_predicted,h)
    R_F7,R_F8=AQI_without_Domain_Knowledge_Future_Ridge(data, y.AQI_predicted,h)

##using Lasso
    data=pd.read_csv('Original_with_dummies.csv')
    y=data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI']=y
    h=BestParams(data)
    L1,L2=AQI_Lasso(data,h)
    L3,L4=AQI_Feature_importance_Lasso(data,h)
    L5,L6=AQI_Domain_Knowledge_Lasso(data,h)
    L7,L8=AQI_without_Domain_Knowledge_Lasso(data,h)

## Predincting for nxt day
    data=pd.read_csv('Original_with_dummies.csv')
    normalize(data)
    h=BestParams(data)
    y=pd.read_csv('AQI_prediction_add.csv')
    L_F1,L_F2=AQI_Future_Lasso(data,y.AQI_predicted,h)
    L_F3,L_F4=AQI_Feature_importance_Future_Lasso(data,y.AQI_predicted,h)
    L_F5,L_F6=AQI_Domain_Knowledge_Future_Lasso(data,y.AQI_predicted,h)
    L_F7,L_F8=AQI_without_Domain_Knowledge_Future_Lasso(data,y.AQI_predicted,h)





##Random forest
    #All feautres
    data = pd.read_csv('Original_with_dummies.csv')
    y = data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI'] = y
    F1,F2=AQI_RF(data)
    F3,F4=AQI_Feature_importance_RF(data)
    F5,F6=AQI_Domain_Knowledge_RF(data)
    F7,F8=AQI_without_Domain_Knowledge_RF(data)

    ## Predincting for nxt day
    data = pd.read_csv('Original_with_dummies.csv')
    normalize(data)
    y = pd.read_csv('AQI_prediction_add.csv')
    F_F1,F_F2=AQI_Future_RF(data, y.AQI_predicted)
    F_F3,F_F4=AQI_Feature_importance_Future_RF(data, y.AQI_predicted)
    F_F5,F_F6=AQI_Domain_Knowledge_Future_RF(data, y.AQI_predicted)
    F_F7,F_F8=AQI_without_Domain_Knowledge_Future_RF(data, y.AQI_predicted)

##NN
    data=pd.read_csv('Original_with_dummies.csv')
    y = data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI'] = y
    layer = [4,4,4]
    NN1,NN2=AQI_NN(data, layer)
    NN3,NN4=AQI_Feature_importance_NN(data, layer)
    NN5,NN6=AQI_Domain_Knowledge_NN(data, layer)
    NN7,NN8=AQI_without_Domain_Knowledge_NN(data, layer)

    ## Predincting for nxt day
    data=pd.read_csv('Original_with_dummies.csv')
    y=pd.read_csv('AQI_prediction_add.csv')
    normalize(data)
    NN_F1,NN_F2=AQI_Future_NN(data,y.AQI_predicted, layer)
    NN_F3,NN_F4=AQI_Feature_importance_Future_NN(data,y.AQI_predicted,layer)
    NN_F5,NN_F6=AQI_Domain_Knowledge_Future_NN(data,y.AQI_predicted,layer)
    NN_F7,NN_F8=AQI_without_Domain_Knowledge_Future_NN(data,y.AQI_predicted, layer)

##All features v/s all models
    Bar_graph (LR1,LR2,L1,L2,R1,R2,S1,S2,F1,F2,NN1,NN2)
##iMPORTANT FEATURES V/S ALL MODELS
    Bar_graph (LR3,LR4,L3,L4,R3,R4,S3,S4,F3,F4,NN3,NN4)
##Future with important features V/S ALL MODELS except svm
    Bar_graph_without_svm (LR_F3,LR_F4,L_F3,L_F4,R_F3,R_F4,F_F3,F_F4,NN_F3,NN_F4)
##Autumn winter and summer
    Bar_graph_season (LR_A3,LR_A4,LR_S3,LR_S4,LR_W3,LR_W4)

##Best Model Analysis using Data
    data = pd.read_csv('Original_with_dummies.csv')
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    train=90
    test=18
    tips=[]
    LABELS=[]
    d=[0,1,2,3,4,5,6,7,8,9]
    for i in range (10):
        train=train+30
        test=test+6
        LABELS.append(train)
        tips.append(train_test_data_prepare(data, train, test, 15))
    plt.plot(tips)
    plt.xticks(d, LABELS)
    plt.xlabel("No of Days")
    plt.ylabel("RMSE")
    plt.title("Models")
    plt.legend()
    plt.show()


#Predicting AQI using all features
def AQI(data):
    y=data.AQI
    data=data.drop('AQI',axis=1)
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test



#Predicting AQI using features from features importance graph
def AQI_Feature_importance(data):
    tree_clf = ExtraTreesRegressor()
    y=data['AQI']
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_Domain_Knowledge(data):
    y=data.AQI
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area',]]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def AQI_without_Domain_Knowledge(data):
    y=data.AQI
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test



def AQI_Future(data,y):
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_Future(data,y):
    tree_clf = ExtraTreesRegressor()
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_Domain_Knowledge_Future(data,y):
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area','month_10','month_11',]]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def AQI_without_Domain_Knowledge_Future(data,y):
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)
    data=data.drop('month_10',axis=1)
    data=data.drop('month_11',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test




def graph_training(y_pred,y_train):
    all_samples = [i for i in range(0, 250)]
    y_pred=y_pred[0:250]
    y_train=y_train[0:250]
    plt.plot(all_samples, y_pred,label='Predicted')
    plt.plot(all_samples , y_train,label='Expected')
    plt.xlabel("No of Samples")
    plt.ylabel("AQI")
    plt.title("Training")
    plt.legend()
    plt.show()


def graph_testing(y_pred,y_val):
    all_samples = [i for i in range(0, 250)]
    y_pred=y_pred[0:250]
    y_val=y_val[0:250]
    plt.plot(all_samples, y_pred,label='Predicted')
    plt.plot(all_samples , y_val,label='Expected')
    plt.xlabel("No of Samples")
    plt.ylabel("AQI")
    plt.title("Validation")
    plt.legend()
    plt.show()




## svm

def AQI_SVM(data):
    y=data.AQI
    data=data.drop('AQI',axis=1)
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = SVR(gamma='scale')
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_SVM(data):
    tree_clf = ExtraTreesRegressor()
    y=data['AQI']
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = SVR(gamma='scale')
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_Domain_Knowledge_SVM(data):
    y=data.AQI
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    # df[['Name', 'Qualification']]
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area',]]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = SVR(gamma='scale')
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def AQI_without_Domain_Knowledge_SVM(data):
    y=data.AQI
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)
    # data=data.drop('month_10',axis=1)
    # data=data.drop('month_11',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = SVR(gamma='scale')
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


def BestParams(data):
    y = data.AQI
    data = data.drop('AQI', axis=1)
    Hyper_params = np.array(
        [ 0.011, 0.1, 0.001, 0.01,.3, .2, 0.6, .8,  0.001, 0.0001, 3, 4,1,2.4])


    Reg_model = Ridge()
    GSCgrid = gsc(estimator=Reg_model, param_grid=dict(alpha=Hyper_params))
    GSCgrid.fit(data, y)
    # print('Hyper Parameter for Ridge:', GSCgrid.best_estimator_.alpha)
    return GSCgrid.best_estimator_.alpha
def normalize(data):
    for c in data.columns:
        mean = data[c].mean()
        max = data[c].max()
        min = data[c].min()
        data[c] = (data[c] - min) / (max - min)
    return data




def AQI_Ridge(data,h):
    y=data.AQI
    data=data.drop('AQI',axis=1)
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = Ridge(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_Ridge(data,h):
    tree_clf = ExtraTreesRegressor()
    y=data['AQI']
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = Ridge(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_Domain_Knowledge_Ridge(data,h):
    y=data.AQI
    # df[['Name', 'Qualification']]
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area']]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = Ridge(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def AQI_without_Domain_Knowledge_Ridge(data,h):
    y=data.AQI
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)


    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = Ridge(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def AQI_Future_Ridge(data,y,h):
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = Ridge(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_Future_Ridge(data,y,h):
    tree_clf = ExtraTreesRegressor()
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = Ridge(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_Domain_Knowledge_Future_Ridge(data,y,h):
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area']]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = Ridge(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def AQI_without_Domain_Knowledge_Future_Ridge(data,y,h):
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)


    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = Ridge(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_Lasso(data,h):
    y=data.AQI
    data=data.drop('AQI',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = Lasso(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_Lasso(data,h):
    tree_clf = ExtraTreesRegressor()
    y=data['AQI']
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = Lasso(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_Domain_Knowledge_Lasso(data,h):
    y=data.AQI
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area']]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = Lasso(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def AQI_without_Domain_Knowledge_Lasso(data,h):
    y=data.AQI
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr =Lasso(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test



def AQI_Future_Lasso(data,y,h):
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr =Lasso(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_Future_Lasso(data,y,h):
    tree_clf = ExtraTreesRegressor()
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = Lasso(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_Domain_Knowledge_Future_Lasso(data,y,h):
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area']]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = Lasso(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def AQI_without_Domain_Knowledge_Future_Lasso(data,y,h):
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = Lasso(alpha=h)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def AQI_RF(data):
    y=data.AQI
    data=data.drop('AQI',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_RF(data):
    tree_clf = ExtraTreesRegressor()
    y=data['AQI']
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_Domain_Knowledge_RF(data):
    y=data.AQI
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area']]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def AQI_without_Domain_Knowledge_RF(data):
    y=data.AQI
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr =RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test



def AQI_Future_RF(data,y):
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_Future_RF(data,y):
    tree_clf = ExtraTreesRegressor()
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_Domain_Knowledge_Future_RF(data,y):
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area']]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def AQI_without_Domain_Knowledge_Future_RF(data,y):
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)


    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_NN(data,layer):
    y=data.AQI
    data=data.drop('AQI',axis=1)
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = MLPRegressor(hidden_layer_sizes=(layer),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01,
                                       batch_size=500,
                                        # early_stopping=True,
                                       random_state=1)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_NN(data, layer):
    tree_clf = ExtraTreesRegressor()
    y=data['AQI']
    data=data.drop('AQI',axis=1)
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr =  MLPRegressor(hidden_layer_sizes=(layer),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01,
                                       batch_size=500,
                                        # early_stopping=True,
                                       random_state=1)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

#Predicting AQI using all features
def AQI_Domain_Knowledge_NN(data, layer):
    y=data.AQI
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area']]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr =  MLPRegressor(hidden_layer_sizes=(layer),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01,
                                       batch_size=500,
                                        # early_stopping=True,
                                       random_state=1)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


def AQI_without_Domain_Knowledge_NN(data,layer):
    y=data.AQI
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = MLPRegressor(hidden_layer_sizes=(layer),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01,
                                       batch_size=500,
                                        # early_stopping=True,
                                       random_state=1)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test



def AQI_Future_NN(data,y, layer):
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = MLPRegressor(hidden_layer_sizes=(layer),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01,
                                       batch_size=500,
                                        # early_stopping=True,
                                       random_state=1)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_Future_NN(data,y, layer):
    tree_clf = ExtraTreesRegressor()
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr =  MLPRegressor(hidden_layer_sizes=(layer),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01,
                                       batch_size=500,
                                        # early_stopping=True,
                                       random_state=1)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


#Predicting AQI using all features
def AQI_Domain_Knowledge_Future_NN(data,y,layer):
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area']]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = MLPRegressor(hidden_layer_sizes=(layer),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01,
                                       batch_size=500,
                                        # early_stopping=True,
                                       random_state=1)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test


def AQI_without_Domain_Knowledge_Future_NN(data,y, layer):
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)


    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr =  MLPRegressor(hidden_layer_sizes=(layer),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01,
                                       batch_size=500,
                                        # early_stopping=True,
                                       random_state=1)
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    train= np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    test=np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    graph_testing(y_pred,y_val)
    return train,test

def Bar_graph (a1,a2,b1,b2,c1,c2,d1,d2,e1,e2,f1,f2):
    barWidth = 0.2
    bars2 = [a2,b2,c2,d2,e2,f2]
    bars1 = [a1,b1,c1,d1,e1,f1]
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black',  capsize=7, label='Train')
    plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black',  capsize=7, label='Test')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['LinearRegression', 'LR with Lasso','LR with Ridge','SVM','random forest', 'Neural Network'])
    plt.ylabel('RMSE')
    plt.xlabel('Models')
    plt.legend()
    plt.show()

def Bar_graph_without_svm(a1,a2,b1,b2,c1,c2,d1,d2,e1,e2):
    barWidth = 0.2
    bars2 = [a2,b2,c2,d2,e2]
    bars1 = [a1,b1,c1,d1,e1]
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black',  capsize=7, label='Train')
    plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black',capsize=7, label='Test')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['LinearRegression', 'LR with Lasso','LR with Ridge','random forest', 'Neural Network'])
    plt.ylabel('RMSE')
    plt.xlabel('Models')
    plt.legend()
    plt.show()

def Bar_graph_season(a1,a2,b1,b2,c1,c2):
    barWidth = 0.2
    bars2 = [a2,b2,c2]
    bars1 = [a1,b1,c1]
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black',  capsize=7, label='Train')
    plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black',capsize=7, label='Test')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Autumn', 'Summer','Winter'])
    plt.ylabel('RMSE')
    plt.xlabel('Seasons')
    plt.legend()
    plt.show()


def train_test_data_prepare(data, train, test, folds):
    d_y = pd.read_csv('AQI_prediction_add.csv')
    y = d_y.AQI_predicted
    x_data = []
    y_data = []
    errors = []
    for i in range(folds):

        x_train = data.loc[i*(train+test):(i*(train+test)+train - 1), :]
        x_test = data.loc[(i*(train+test)+train):(i+1)*(train+test)-1, :]
        y_train = y.loc[i * (train + test):(i * (train + test) + train - 1)]
        y_test = y.loc[(i * (train + test) + train):(i + 1) * (train + test) - 1]
        regr = MLPRegressor(hidden_layer_sizes=(4, 4),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01,
                                       # batch_size=500,
                                        # early_stopping=True,
                                       random_state=1)
        regr.fit(x_train, y_train)
        print("xxxx")
        y_pred = regr.predict(x_train)
        print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
        y_pred = regr.predict(x_test)
        print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        errors.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    print("Cross validation test error = ", sum(errors)/len(errors))
    return sum(errors)/len(errors)








main()
