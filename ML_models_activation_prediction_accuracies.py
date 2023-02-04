import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score,cross_val_predict
from xgboost import XGBClassifier, XGBRegressor
import xgboost
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit

train = np.load('Training_features_in_555_proteins.npy', allow_pickle=True)
df_train = pd.DataFrame(train)
train_data = df_train.iloc[:, 3:]
train_act = df_train.iloc[:,1]
train_state = df_train.iloc[:,2]


# #. RF

def model_prediction_RandomForestClassifier(data_inp, label_inp):
    clf = RandomForestClassifier(n_estimators=18,random_state = 25)    
    scores = cross_val_score(clf, data_inp, label_inp, cv=5, scoring='accuracy')
    clf.fit(data_inp,label_inp)
    return scores


pred_rf  = model_prediction_RandomForestClassifier(train_data, train_state.astype('int'))
print("RF State_prediction: ",format(pred_rf.mean()*100,'.2f'), "% accuracy with SD ",  format(pred_rf.std(),'.2f') )


def model_RandomForestRegressor(data_inp, label_inp):
    clf = RandomForestRegressor()
    clf.fit(data_inp,label_inp)
    scores = cross_val_score(clf, data_inp, label_inp, cv=5, scoring='neg_mean_absolute_error')
    return scores


pred_reg  = model_RandomForestRegressor(train_data, train_act.astype('int'))
print("RF regressor_prediction: ",format(pred_reg.mean(),'.2f'), "accuracy with SD ",  format(pred_reg.std(),'.2f') )


# #. XGBoost

def model_prediction_XGBoostClassifier(data_inp, label_inp):
    xgb_model = xgboost.XGBClassifier(num_class=10,
                                  learning_rate=0.7,
                                  max_depth=10, 
                                  use_label_encoder=False,
                                  eval_metric='mlogloss')
    xgb_model.fit(data_inp.astype('float').values,label_inp.astype('int').values)
    scores = cross_val_score(xgb_model, np.array(data_inp.values), label_inp, cv=5, scoring='accuracy')
    return scores

pred_xg  = model_prediction_XGBoostClassifier(train_data, train_state.astype('int'))
print("XGB State_prediction: ",format(pred_xg.mean()*100,'.2f'), "% accuracy with SD ",  format(pred_xg.std(),'.2f') )


def model_prediction_XGBoostreg(data_inp, label_inp):
    rgr_xgb = XGBRegressor(use_label_encoder=False, eval_metric='mlogloss')
    rgr_xgb.fit(data_inp.astype('float').values,label_inp.astype('int').values)
    scores = cross_val_score(rgr_xgb, np.array(data_inp.values), label_inp, cv=5, scoring='neg_mean_absolute_error')
    return scores


reg_xg  = model_prediction_XGBoostreg(train_data, train_act.astype('int'))
print("XGB regressor_prediction: ",format(reg_xg.mean(),'.2f'), "accuracy with SD ",  format(reg_xg.std(),'.2f') )


# #. DT

def model_prediction_DTClassifier(data_inp, label_inp):
    clf = clf = DecisionTreeClassifier(random_state=42)
    clf.fit(data_inp,label_inp)    
    scores = cross_val_score(clf, data_inp, label_inp, cv=5, scoring='accuracy')
    return scores

pred_dt  = model_prediction_DTClassifier(train_data, train_state.astype('int'))
print("DT State_prediction: ",format(pred_dt.mean()*100,'.2f'), "% accuracy with SD ",  format(pred_dt.std(),'.2f') )


def model_prediction_DTreg(data_inp, label_inp):
    clf = clf = DecisionTreeRegressor(max_depth=2)
    clf.fit(data_inp,label_inp)    
    scores = cross_val_score(clf, data_inp, label_inp, cv=5, scoring='neg_mean_absolute_error')
    return scores


reg_dt  = model_prediction_DTreg(train_data, train_act.astype('int'))
print("DT regressor_prediction: ",format(reg_dt.mean(),'.2f'), "accuracy with SD ",  format(reg_dt.std(),'.2f') )

