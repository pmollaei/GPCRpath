#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score,cross_val_predict
from xgboost import XGBClassifier, XGBRegressor
import xgboost
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit


# In[2]:


train = np.load('asn_pro_tyr_angles_55_cont_pn_555_prot.npy', allow_pickle=True)
df_train = pd.DataFrame(train)
train_data = df_train.iloc[:, 3:]
train_act = df_train.iloc[:,1]
train_state = df_train.iloc[:,2]


# #. RF

# In[3]:


def model_prediction_RandomForestClassifier(data_inp, label_inp):
    clf = RandomForestClassifier(n_estimators=18,random_state = 25)    
    scores = cross_val_score(clf, data_inp, label_inp, cv=5, scoring='accuracy')
    clf.fit(data_inp,label_inp)
    return scores


# In[4]:


pred_rf  = model_prediction_RandomForestClassifier(train_data, train_state.astype('int'))
print("RF State_prediction: ",format(pred_rf.mean()*100,'.2f'), "% accuracy with SD ",  format(pred_rf.std(),'.2f') )


# In[5]:


def model_RandomForestRegressor(data_inp, label_inp):
    clf = RandomForestRegressor()
    clf.fit(data_inp,label_inp)
    scores = cross_val_score(clf, data_inp, label_inp, cv=5, scoring='neg_mean_absolute_error')
    return scores


# In[6]:


pred_reg  = model_RandomForestRegressor(train_data, train_act.astype('int'))
print("RF regressor_prediction: ",format(pred_reg.mean(),'.2f'), "accuracy with SD ",  format(pred_reg.std(),'.2f') )


# #. XGBoost

# In[7]:


def model_prediction_XGBoostClassifier(data_inp, label_inp):
    xgb_model = xgboost.XGBClassifier(num_class=10,
                                  learning_rate=0.7,
                                  max_depth=10, 
                                  use_label_encoder=False,
                                  eval_metric='mlogloss')
    xgb_model.fit(data_inp.astype('float').values,label_inp.astype('int').values)
    scores = cross_val_score(xgb_model, np.array(data_inp.values), label_inp, cv=5, scoring='accuracy')
    return scores


# In[15]:


pred_xg  = model_prediction_XGBoostClassifier(train_data, train_state.astype('int'))
print("XGB State_prediction: ",format(pred_xg.mean()*100,'.2f'), "% accuracy with SD ",  format(pred_xg.std(),'.2f') )


# In[16]:


def model_prediction_XGBoostreg(data_inp, label_inp):
    rgr_xgb = XGBRegressor(use_label_encoder=False, eval_metric='mlogloss')
    rgr_xgb.fit(data_inp.astype('float').values,label_inp.astype('int').values)
    scores = cross_val_score(rgr_xgb, np.array(data_inp.values), label_inp, cv=5, scoring='neg_mean_absolute_error')
    return scores


# In[17]:


reg_xg  = model_prediction_XGBoostreg(train_data, train_act.astype('int'))
print("XGB regressor_prediction: ",format(reg_xg.mean(),'.2f'), "accuracy with SD ",  format(reg_xg.std(),'.2f') )


# #. DT

# In[11]:


def model_prediction_DTClassifier(data_inp, label_inp):
    clf = clf = DecisionTreeClassifier(random_state=42)
    clf.fit(data_inp,label_inp)    
    scores = cross_val_score(clf, data_inp, label_inp, cv=5, scoring='accuracy')
    return scores


# In[12]:


pred_dt  = model_prediction_DTClassifier(train_data, train_state.astype('int'))
print("DT State_prediction: ",format(pred_dt.mean()*100,'.2f'), "% accuracy with SD ",  format(pred_dt.std(),'.2f') )


# In[13]:


def model_prediction_DTreg(data_inp, label_inp):
    clf = clf = DecisionTreeRegressor(max_depth=2)
    clf.fit(data_inp,label_inp)    
    scores = cross_val_score(clf, data_inp, label_inp, cv=5, scoring='neg_mean_absolute_error')
    return scores


# In[14]:


reg_dt  = model_prediction_DTreg(train_data, train_act.astype('int'))
print("DT regressor_prediction: ",format(reg_dt.mean(),'.2f'), "accuracy with SD ",  format(reg_dt.std(),'.2f') )

