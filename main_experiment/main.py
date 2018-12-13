# -*- coding: utf-8 -*-


import numpy as np
import xgboost as xgbxgb
import pandas as pd
from sklearn import cross_validation
import config
import math
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
import kmediods
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
import sys
def computer_RRMSE_list(real_test,result_p,real_train_mean):
    _list = []
    for i in range(result_p.shape[1]):
        fenzi = 0
        fenmu = 0
        for j in range(result_p.shape[0]):
            fenzi += (result_p[j,i] - real_test[j,i])**2
            fenmu += (real_train_mean[i] - real_test[j,i])**2
        _list.append(math.sqrt(fenzi/fenmu))    
    return _list

def model_xgb(x_train,y_train,x_test,y_test,booster,eta,max_depth,colsample_bytree,subsample,min_child_weight,reg_lambda,num_boost):
    dtrain = xgbxgb.DMatrix(x_train, label=y_train)
    dtest = xgbxgb.DMatrix(x_test)
    # 模型参数
    params = {'booster': booster,
              'objective':'reg:linear',
              'eta': eta,
              'max_depth': max_depth,  # 4 3
              'colsample_bytree': colsample_bytree,#0.8
              'subsample': subsample,
              'min_child_weight': min_child_weight  # 2 3
              ,'reg_lambda':reg_lambda
              }
#    print(booster,eta,max_depth,colsample_bytree,subsample,min_child_weight,reg_lambda)
    bst = xgbxgb.train(params, dtrain, num_boost_round=num_boost)
    predict = bst.predict(dtest)
    return predict

def model_r(x_train,y_train,x_test,y_test,booster,eta,max_depth,colsample_bytree,subsample,min_child_weight,reg_lambda,num_boost,my_num):
    bst = RandomForestRegressor(n_estimators = my_num,n_jobs = -1).fit(x_train,y_train)
    predict = bst.predict(x_test)
    return predict

def model_svr(x_train,y_train,x_test,y_test,ker,gam,c,epsi,shrink,verb,max_it):#90
    bst = SVR(kernel=ker, gamma=gam, C=c, epsilon=epsi, shrinking=shrink, verbose=verb, max_iter=max_it).fit(x_train,y_train)
    predict = bst.predict(x_test)
    return predict

def get_linjie_matrix(x_train,y_train):
    all_all_num = []
    for rr in range(y_train.shape[1]):
        delta_loss = y_train[:,rr].copy()
        all_result = []
        all_weight= []
        num_epch = config.all_config[text_file].get('adjacency_num')
        for qq in range(num_epch):
            estimator = DecisionTreeRegressor(max_depth=config.all_config[text_file].get('adjacency_cart_depth')).fit(x_train,delta_loss)
            leave_id = estimator.apply(x_train)
            result = []
            for i in range(len(leave_id)):
                temp_one = []
                for j in range(len(leave_id)):
                    if (leave_id[i] == leave_id[j]):
                        temp_one.append(1)
                    else:
                        temp_one.append(0)
                result.append(temp_one)
            result = np.array(result)
            
            pre = estimator.predict(x_train)
            delta_loss = delta_loss-pre
            all_weight.append(sum(abs(delta_loss)))
            all_result.append(result)
        all_num = (np.sum(all_weight)-all_weight[0])/np.sum(all_weight)*all_result[0]
        for qq in range(1,num_epch):
            all_num += (np.sum(all_weight)-all_weight[qq])/np.sum(all_weight)*all_result[qq]
        all_all_num.append(all_num)
    cos_result = target_cos(y_train)
    return all_all_num,cos_result

def select_feature(x_train,y_train,x_test,which_col,cos_result,all_all_num):
    all_num = np.zeros((y_train.shape[0], y_train.shape[0]))
    for rr in range(len(cos_result)):
        all_num+=cos_result[rr,which_col]*all_all_num[rr]
    all_num =-(all_num- all_num[0,0])
    num_cluster = config.all_config[text_file].get('num_cluster')
    M, C = kmediods.kMedoids(all_num, num_cluster,config.all_config[text_file].get('cluster_times'))
    train_add = []
    test_add = []
    for k in range(num_cluster):
        result_k = np.apply_along_axis(lambda x:sum(abs(x-x_train[M[k]])),1,x_train)
        result_k = np.array([result_k]).T
        train_add.append(result_k)
        result_test_k = np.apply_along_axis(lambda x:sum(abs(x-x_train[M[k]])),1,x_test)
        result_test_k = np.array([result_test_k]).T
        test_add.append(result_test_k)

    for k in range(num_cluster):
        x_train = np.concatenate((x_train,train_add[k]),axis=1)
        x_test = np.concatenate((x_test,test_add[k]),axis=1)
    return x_train,x_test


def target_cos(y_train):
    result = np.zeros((y_train.shape[1], y_train.shape[1]))
    for i in range(y_train.shape[1]):
        for j in range(y_train.shape[1]):
            result[i,j] = sum(y_train[:,i]*y_train[:,j])/(math.sqrt(sum(np.power(y_train[:,i],2))) * math.sqrt(sum(np.power(y_train[:,j],2))) + 0.001)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] = np.exp(config.all_config[text_file].get('target_lambda')*(1-result[i,j]))
    return result


if __name__=='__main__':
#    text_file = 'atp1d'
    text_file = sys.argv[1]
    data = pd.read_csv(r'../data/'+text_file+'.arff',header = config.all_config[text_file].get('header') -4)
    data.reset_index(inplace=True)
    data.replace('?',np.nan,inplace=True)
    data.replace('     ?',np.nan,inplace=True)
    data = data.applymap(float)
    
    
    if config.all_config[text_file].get('sample_random') == True:
        data = data.sample(frac=1,random_state = config.all_config[text_file].get('sample_random_num')).reset_index(drop=True)
    data = data.fillna(pd.Series(np.nanmean(data,axis=0),index=data.columns))
    label = data.iloc[:,-config.all_config[text_file].get('targets_num'):].values
    data = data.iloc[:,:-config.all_config[text_file].get('targets_num')].values
    
    
    if config.all_config[text_file].get('is_Agg_clustering') == True:
        clustering = AgglomerativeClustering(n_clusters=config.all_config[text_file].get('Agg_clustering_num')).fit(data)
        data = np.concatenate((data,np.array([clustering.labels_]).T),axis=1)
    
    loss_list_RRMSE_list = []
    kf = KFold(n_splits=config.all_config[text_file].get('paper_kFold'),random_state=config.all_config[text_file].get('kflod_random_num'),shuffle=config.all_config[text_file].get('kflod_random'))
    for train_index, test_index in kf.split(data):
        s_x_train, s_x_test = data[train_index], data[test_index]
        s_y_train, s_y_test = label[train_index], label[test_index]
        result_p_train_1_stage = []
        result_p_test_1_stage = []
        if config.all_config[text_file].get('tift') == True:
            all_all_num,cos_result = get_linjie_matrix(s_x_train,s_y_train)
        for i in range(label.shape[1]):
            if config.all_config[text_file].get('tift') == True:
                x_train,x_test = select_feature(s_x_train,s_y_train,s_x_test,i,cos_result,all_all_num)
            else:
                x_train,x_test = s_x_train.copy(),s_x_test.copy()
            r1_list = []
            kf_2 = KFold(n_splits=config.all_config[text_file].get('inner_kFold'))
            for train_index_2, test_index_2 in kf_2.split(x_train):
                x_train_2, x_test_2 = x_train[train_index_2], x_train[test_index_2]
                y_train_2, y_test_2 = s_y_train[train_index_2], s_y_train[test_index_2]
                r1 = None
                if text_file in ['sf1','sf2']:
                    r1 = model_svr(x_train_2,y_train_2[:,i],x_test_2,y_test_2[:,i],
                                   config.all_config[text_file].get('svr1_kernel'),
                                   config.all_config[text_file].get('svr1_gamma'),
                                   config.all_config[text_file].get('svr1_C'),
                                   config.all_config[text_file].get('svr1_epsilon'),
                                   config.all_config[text_file].get('svr1_shrinking'),
                                   config.all_config[text_file].get('svr1_verbose'),
                                   config.all_config[text_file].get('svr1_max_iter'))
                else:
                    r1 = model_xgb(x_train_2,y_train_2[:,i],x_test_2,y_test_2[:,i],
                                   config.all_config[text_file].get('xgb1_booster'),
                                   config.all_config[text_file].get('xgb1_eta'),
                                   config.all_config[text_file].get('xgb1_max_depth'),
                                   config.all_config[text_file].get('xgb1_colsample_bytree'),
                                   config.all_config[text_file].get('xgb1_subsample'),
                                   config.all_config[text_file].get('xgb1_min_child_weight'),
                                   config.all_config[text_file].get('xgb1_reg_lambda',0),
                                   config.all_config[text_file].get('xgb1_boost_round'))
                r1_list.append(r1)
            result_p_train_1_stage.append(np.concatenate(r1_list,axis=0))
            predicted_test = None
            if text_file in ['sf1','sf2']:
                predicted_test = model_svr(x_train_2,y_train_2[:,i],x_test_2,y_test_2[:,i],
                                   config.all_config[text_file].get('svr1_kernel'),
                                   config.all_config[text_file].get('svr1_gamma'),
                                   config.all_config[text_file].get('svr1_C'),
                                   config.all_config[text_file].get('svr1_epsilon'),
                                   config.all_config[text_file].get('svr1_shrinking'),
                                   config.all_config[text_file].get('svr1_verbose'),
                                   config.all_config[text_file].get('svr1_max_iter'))
            else:
                predicted_test = model_xgb(x_train,s_y_train[:,i],x_test,None,
                                           config.all_config[text_file].get('xgb1_booster'),
                                           config.all_config[text_file].get('xgb1_eta'),
                                           config.all_config[text_file].get('xgb1_max_depth'),
                                           config.all_config[text_file].get('xgb1_colsample_bytree'),
                                           config.all_config[text_file].get('xgb1_subsample'),
                                           config.all_config[text_file].get('xgb1_min_child_weight'),
                                           config.all_config[text_file].get('xgb1_reg_lambda',0),
                                           config.all_config[text_file].get('xgb1_boost_round'))
            result_p_test_1_stage.append(predicted_test)
        result_p_train_1_stage = pd.DataFrame(result_p_train_1_stage).T.values
        result_p_test_1_stage = pd.DataFrame(result_p_test_1_stage).T.values
        new_train = np.concatenate((x_train,result_p_train_1_stage),axis=1)
        new_test = np.concatenate((x_test,result_p_test_1_stage),axis=1)
        if config.all_config[text_file].get('is_feature_selection',False)==True:
            new_train  = new_train[:,:config.all_config[text_file].get('feature_selection')]
            new_test  = new_test[:,:config.all_config[text_file].get('feature_selection')]
        result_p_test_2_stage = []
        for i in range(label.shape[1]):
            predicted_test = None
            if text_file in ['sf1','sf2']:
                predicted_test = model_svr(x_train_2,y_train_2[:,i],x_test_2,y_test_2[:,i],
                               config.all_config[text_file].get('svr2_kernel'),
                               config.all_config[text_file].get('svr2_gamma'),
                               config.all_config[text_file].get('svr2_C'),
                               config.all_config[text_file].get('svr2_epsilon'),
                               config.all_config[text_file].get('svr2_shrinking'),
                               config.all_config[text_file].get('svr2_verbose'),
                               config.all_config[text_file].get('svr2_max_iter'))
            else:
                tmp_i = i if i>=2 else 2
                predicted_test = model_xgb(new_train,s_y_train[:,i],new_test,None, 
                                           config.all_config[text_file].get('xgb'+str(tmp_i)+'_booster',config.all_config[text_file].get('xgb2_booster')),
                                           config.all_config[text_file].get('xgb'+str(tmp_i)+'_eta',config.all_config[text_file].get('xgb2_eta')),
                                           config.all_config[text_file].get('xgb'+str(tmp_i)+'_max_depth',config.all_config[text_file].get('xgb2_max_depth')),
                                           config.all_config[text_file].get('xgb'+str(tmp_i)+'_colsample_bytree',config.all_config[text_file].get('xgb2_colsample_bytree')),
                                           config.all_config[text_file].get('xgb'+str(tmp_i)+'_subsample',config.all_config[text_file].get('xgb2_subsample')),
                                           config.all_config[text_file].get('xgb'+str(tmp_i)+'_min_child_weight',config.all_config[text_file].get('xgb2_min_child_weight')),
                                           config.all_config[text_file].get('xgb'+str(tmp_i)+'_reg_lambda',config.all_config[text_file].get('xgb2_reg_lambda',0)),
                                           config.all_config[text_file].get('xgb'+str(tmp_i)+'_boost_round',config.all_config[text_file].get('xgb2_boost_round')))
                
            result_p_test_2_stage.append(predicted_test)       
        result_p_test_2_stage = pd.DataFrame(result_p_test_2_stage).T.values
        real_train_mean = s_y_train.mean(axis=0)
        loss_list_RRMSE_list.append(computer_RRMSE_list(s_y_test,result_p_test_2_stage,real_train_mean))
    loss_list_RRMSE_list_np = np.array(loss_list_RRMSE_list)
    print(loss_list_RRMSE_list_np.mean(axis=0))
    print(np.mean(loss_list_RRMSE_list_np.mean(axis=0)))