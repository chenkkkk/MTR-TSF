# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 09:05:54 2018
@author: CCL
"""

andro_ = {'xgb1_booster':'gbtree','xgb1_eta':0.028,'xgb1_max_depth': 3,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.7,'xgb1_min_child_weight': 4,'xgb1_boost_round':520,
          'xgb2_booster':'gbtree','xgb2_eta':0.031,'xgb2_max_depth': 3,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.7,'xgb2_min_child_weight': 6,'xgb2_boost_round':520,
          'adjacency_num':20,'num_cluster':3,'cluster_times':20,'target_lambda':-5,'targets_num':6,'is_Agg_clustering':True,'Agg_clustering_num':2,'paper_kFold':10,'inner_kFold':5, 
          'adjacency_cart_depth':5,'kflod_random':False,'kflod_random_num':-1,'sample_random':True,'sample_random_num':2,'header':41,'tift':True}

atp1d_ = {'xgb1_booster':'gbtree','xgb1_eta':0.03,'xgb1_max_depth': 3,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.7,'xgb1_min_child_weight': 5,'xgb1_boost_round':300,
          'xgb2_booster':'gbtree','xgb2_eta':0.03,'xgb2_max_depth': 3,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.7,'xgb2_min_child_weight': 8,'xgb2_boost_round':300,
          'adjacency_num':20,'num_cluster':6,'cluster_times':20,'target_lambda':-5,'targets_num':6,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':5, 
          'adjacency_cart_depth':3,'kflod_random':True,'kflod_random_num':0,'sample_random':False,'sample_random_num':-1,'header':422,'tift':True}

atp7d_ = {'xgb1_booster':'gbtree','xgb1_eta':0.03,'xgb1_max_depth': 3,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.7,'xgb1_min_child_weight': 3,'xgb1_boost_round':1000,
          'xgb2_booster':'gbtree','xgb2_eta':0.02,'xgb2_max_depth': 3,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.7,'xgb2_min_child_weight': 3,'xgb2_boost_round':1000,
          'adjacency_num':20,'num_cluster':5,'cluster_times':20,'target_lambda':-5,'targets_num':6,'is_Agg_clustering':True,'Agg_clustering_num':4,'paper_kFold':10,'inner_kFold':10,
          'adjacency_cart_depth':3,'kflod_random':False,'kflod_random_num':-1,'sample_random':True,'sample_random_num':2,'header':422,'tift':True}

edm_ = {'xgb1_booster':'gbtree','xgb1_eta':0.02,'xgb1_max_depth': 4,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.7,'xgb1_min_child_weight': 1,'xgb1_boost_round':600,
          'xgb2_booster':'gbtree','xgb2_eta':0.02,'xgb2_max_depth': 4,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.7,'xgb2_min_child_weight': 3,'xgb2_boost_round':300,
          'adjacency_num':20,'num_cluster':4,'cluster_times':20,'target_lambda':-5,'targets_num':2,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':3,
          'adjacency_cart_depth':3,'kflod_random':False,'kflod_random_num':-1,'sample_random':True,'sample_random_num':12,'header':26,'tift':True}

enb_ = {'xgb1_booster':'gbtree','xgb1_eta':0.02,'xgb1_max_depth': 7,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.6,'xgb1_min_child_weight': 7.5,'xgb1_boost_round':1000,
          'xgb2_booster':'gbtree','xgb2_eta':0.024,'xgb2_max_depth': 6,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.7,'xgb2_min_child_weight': 6,'xgb2_boost_round':1000,'xgb2_reg_lambda':10,
          'adjacency_num':20,'num_cluster':4,'cluster_times':20,'target_lambda':-5,'targets_num':2,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':10,
          'adjacency_cart_depth':3,'kflod_random':False,'kflod_random_num':-1,'sample_random':True,'sample_random_num':0,'header':15,'tift':True}

jura_ = {'xgb1_booster':'gbtree','xgb1_eta':0.02,'xgb1_max_depth': 3,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.6,'xgb1_min_child_weight': 10.5,'xgb1_boost_round':300,
          'xgb2_booster':'gbtree','xgb2_eta':0.02,'xgb2_max_depth': 3,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.6,'xgb2_min_child_weight': 5,'xgb2_boost_round':500,'xgb2_reg_lambda':40,
          'adjacency_num':20,'num_cluster':5,'cluster_times':20,'target_lambda':-5,'targets_num':3,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':5,
          'adjacency_cart_depth':3,'kflod_random':False,'kflod_random_num':-1,'sample_random':True,'sample_random_num':0,'header':23,'tift':True}

oes10_ = {'xgb1_booster':'gbtree','xgb1_eta':0.02,'xgb1_max_depth': 3,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.7,'xgb1_min_child_weight': 2,'xgb1_boost_round':300,
          'xgb2_booster':'gbtree','xgb2_eta':0.02,'xgb2_max_depth': 3,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.7,'xgb2_min_child_weight': 2,'xgb2_boost_round':250,
          'xgb15_booster':'gbtree','xgb15_eta':0.026,'xgb15_max_depth': 4,'xgb15_colsample_bytree': 0.5,'xgb15_subsample': 0.5,'xgb15_min_child_weight': 10,'xgb15_boost_round':150,
          'xgb14_booster':'gblinear','xgb14_eta':0.02,'xgb14_max_depth': 8,'xgb14_colsample_bytree': 0.7,'xgb14_subsample': 0.7,'xgb14_min_child_weight': 0,'xgb14_boost_round':250,
          'xgb13_booster':'gblinear','xgb13_eta':0.02,'xgb13_max_depth': 8,'xgb13_colsample_bytree': 0.7,'xgb13_subsample': 0.7,'xgb13_min_child_weight': 5,'xgb13_boost_round':250,
          'xgb12_booster':'gblinear','xgb12_eta':0.011,'xgb12_max_depth': 8,'xgb12_colsample_bytree': 0.7,'xgb12_subsample': 0.7,'xgb12_min_child_weight': 5,'xgb12_boost_round':250,
          'xgb11_booster':'gblinear','xgb11_eta':0.023,'xgb11_max_depth': 3,'xgb11_colsample_bytree': 0.7,'xgb11_subsample': 0.7,'xgb11_min_child_weight': 5,'xgb11_boost_round':250,
          'xgb10_booster':'gblinear','xgb10_eta':0.013,'xgb10_max_depth': 3,'xgb10_colsample_bytree': 0.7,'xgb10_subsample': 0.7,'xgb10_min_child_weight': 5,'xgb10_boost_round':250,
          'xgb9_booster':'gblinear','xgb9_eta':0.013,'xgb9_max_depth': 3,'xgb9_colsample_bytree': 0.7,'xgb9_subsample': 0.7,'xgb9_min_child_weight': 5,'xgb9_boost_round':250,
          'xgb8_booster':'gblinear','xgb8_eta':0.013,'xgb8_max_depth': 3,'xgb8_colsample_bytree': 0.7,'xgb8_subsample': 0.7,'xgb8_min_child_weight': 5,'xgb8_boost_round':250,
          'xgb7_booster':'gblinear','xgb7_eta':0.013,'xgb7_max_depth': 3,'xgb7_colsample_bytree': 0.7,'xgb7_subsample': 0.7,'xgb7_min_child_weight': 5,'xgb7_boost_round':250,
          'xgb6_booster':'gblinear','xgb6_eta':0.013,'xgb6_max_depth': 3,'xgb6_colsample_bytree': 0.7,'xgb6_subsample': 0.7,'xgb6_min_child_weight': 5,'xgb6_boost_round':250,
          'xgb5_booster':'gblinear','xgb5_eta':0.013,'xgb5_max_depth': 3,'xgb5_colsample_bytree': 0.7,'xgb5_subsample': 0.7,'xgb5_min_child_weight': 5,'xgb5_boost_round':250,
          'xgb4_booster':'gblinear','xgb4_eta':0.013,'xgb4_max_depth': 3,'xgb4_colsample_bytree': 0.7,'xgb4_subsample': 0.7,'xgb4_min_child_weight': 5,'xgb4_boost_round':250,
          'xgb3_booster':'gblinear','xgb3_eta':0.013,'xgb3_max_depth': 3,'xgb3_colsample_bytree': 0.7,'xgb3_subsample': 0.7,'xgb3_min_child_weight': 5,'xgb3_boost_round':250,
          'adjacency_num':20,'num_cluster':3,'cluster_times':20,'target_lambda':-5,'targets_num':16,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':10,
          'adjacency_cart_depth':3,'kflod_random':True,'kflod_random_num':8,'sample_random':False,'sample_random_num':-1,'header':319,'tift':True}



oes97_ = {'xgb1_booster':'gblinear','xgb1_eta':0.017,'xgb1_max_depth': 3,'xgb1_colsample_bytree': 0.5,'xgb1_subsample': 0.7,'xgb1_min_child_weight': 2,'xgb1_boost_round':200,
          'xgb2_booster':'gblinear','xgb2_eta':0.015,'xgb2_max_depth': 3,'xgb2_colsample_bytree': 0.5,'xgb2_subsample': 0.6,'xgb2_min_child_weight': 2,'xgb2_boost_round':370,
          'adjacency_num':20,'num_cluster':4,'cluster_times':20,'target_lambda':-5,'targets_num':16,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':5,
          'adjacency_cart_depth':3,'kflod_random':True,'kflod_random_num':3,'sample_random':False,'sample_random_num':-1,'header':284,'tift':True}



oscales_ = {'xgb1_booster':'gbtree','xgb1_eta':0.03,'xgb1_max_depth': 5,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.7,'xgb1_min_child_weight': 1.2,'xgb1_boost_round':400,
          'xgb2_booster':'gbtree','xgb2_eta':0.03,'xgb2_max_depth': 5,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.7,'xgb2_min_child_weight': 1,'xgb2_boost_round':400,
          'adjacency_num':20,'num_cluster':6,'cluster_times':20,'target_lambda':-5,'targets_num':12,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':5,
          'adjacency_cart_depth':3,'kflod_random':True,'kflod_random_num':2,'sample_random':False,'sample_random_num':-1,'header':418,'tift':True}



scpf_ = {'xgb1_booster':'gbtree','xgb1_eta':0.01,'xgb1_max_depth': 3,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.7,'xgb1_min_child_weight': 5,'xgb1_boost_round':300,
          'xgb2_booster':'gbtree','xgb2_eta':0.01,'xgb2_max_depth': 3,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.7,'xgb2_min_child_weight': 9,'xgb2_boost_round':200,
          'adjacency_num':20,'num_cluster':4,'cluster_times':20,'target_lambda':-5,'targets_num':3,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':3,
          'adjacency_cart_depth':3,'kflod_random':True,'kflod_random_num':9,'sample_random':False,'sample_random_num':-1,'header':31,'tift':True}





sf1_ = {'svr1_kernel':'rbf','svr1_gamma':0.005,'svr1_C':13,'svr1_epsilon':0.0005,'svr1_shrinking':True,'svr1_verbose':False,'svr1_max_iter':-1,
        'svr2_kernel':'rbf','svr2_gamma':0.007,'svr2_C':4,'svr2_epsilon':0.000001,'svr2_shrinking':True,'svr2_verbose':False,'svr2_max_iter':-1,
          'adjacency_num':20,'num_cluster':3,'cluster_times':20,'target_lambda':-5,'targets_num':3,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':10,
          'adjacency_cart_depth':3,'kflod_random':True,'kflod_random_num':3,'sample_random':False,'sample_random_num':-1,'header':18,'tift':True}





sf2_ = {'svr1_kernel':'rbf','svr1_gamma':0.006,'svr1_C':14,'svr1_epsilon':0.005,'svr1_shrinking':True,'svr1_verbose':False,'svr1_max_iter':-1,
        'svr2_kernel':'rbf','svr2_gamma':0.007,'svr2_C':4,'svr2_epsilon':0.000001,'svr2_shrinking':True,'svr2_verbose':False,'svr2_max_iter':-1,
          'adjacency_num':20,'num_cluster':2,'cluster_times':20,'target_lambda':-5,'targets_num':3,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':10,
          'adjacency_cart_depth':3,'kflod_random':True,'kflod_random_num':10,'sample_random':False,'sample_random_num':-1,'header':18,'tift':True}







slump_ = {'svr1_kernel':'rbf','svr1_gamma':0.05,'svr1_C':70,'svr1_epsilon':0.01,'svr1_shrinking':False,'svr1_verbose':False,'svr1_max_iter':-1,
        'svr2_kernel':'rbf','svr2_gamma':0.03,'svr2_C':40,'svr2_epsilon':0.01,'svr2_shrinking':False,'svr2_verbose':False,'svr2_max_iter':-1,
          'adjacency_num':20,'num_cluster':2,'cluster_times':20,'target_lambda':-5,'targets_num':3,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':5,
          'adjacency_cart_depth':3,'kflod_random':True,'kflod_random_num':2,'sample_random':False,'sample_random_num':-1,'header':15,'tift':True}


wq_ = {'xgb1_booster':'gbtree','xgb1_eta':0.02,'xgb1_max_depth': 15,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.6,'xgb1_min_child_weight': 15.5,'xgb1_boost_round':400,
          'xgb2_booster':'gbtree','xgb2_eta':0.03,'xgb2_max_depth': 3,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.6,'xgb2_min_child_weight': 12,'xgb2_boost_round':140,'xgb2_reg_lambda':14,
          'adjacency_num':20,'num_cluster':3,'cluster_times':20,'target_lambda':-5,'targets_num':14,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':10,'inner_kFold':5,
          'adjacency_cart_depth':3,'kflod_random':False,'kflod_random_num':-1,'sample_random':True,'sample_random_num':37,'header':58,'tift':True}


scm1d_ = {'xgb1_booster':'gbtree','xgb1_eta':0.03,'xgb1_max_depth': 14,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.7,'xgb1_min_child_weight': 1,'xgb1_boost_round':500,
          'xgb2_booster':'gbtree','xgb2_eta':0.03,'xgb2_max_depth': 14,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.7,'xgb2_min_child_weight': 1,'xgb2_boost_round':800,
          'adjacency_num':10,'num_cluster':5,'cluster_times':10,'target_lambda':-5,'targets_num':16,'is_Agg_clustering':True,'Agg_clustering_num':2,'paper_kFold':2,'inner_kFold':3,
          'adjacency_cart_depth':3,'kflod_random':False,'kflod_random_num':-1,'sample_random':True,'sample_random_num':2,'is_feature_selection':True,'feature_selection':280,'header':301,'tift':False}




scm20d_ = {'xgb1_booster':'gbtree','xgb1_eta':0.03,'xgb1_max_depth': 5,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.7,'xgb1_min_child_weight': 2,'xgb1_boost_round':500,
          'xgb2_booster':'gbtree','xgb2_eta':0.03,'xgb2_max_depth': 5,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.7,'xgb2_min_child_weight': 2,'xgb2_boost_round':500,
          'adjacency_num':10,'num_cluster':5,'cluster_times':10,'target_lambda':-5,'targets_num':16,'is_Agg_clustering':True,'Agg_clustering_num':2,'paper_kFold':2,'inner_kFold':3,
          'adjacency_cart_depth':3,'kflod_random':False,'kflod_random_num':-1,'sample_random':True,'sample_random_num':2,'is_feature_selection':True,'feature_selection':62,'header':301,'tift':False}



rf1_ = {'my_num':20,
          'xgb1_booster':'gbtree','xgb1_eta':0.2,'xgb1_max_depth': 13,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.6,'xgb1_min_child_weight': 100,'xgb1_boost_round':150,
          'xgb2_booster':'gbtree','xgb2_eta':0.2,'xgb2_max_depth': 13,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.6,'xgb2_min_child_weight': 100,'xgb2_boost_round':150,
          'adjacency_num':10,'num_cluster':5,'cluster_times':10,'target_lambda':-5,'targets_num':8,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':5,'inner_kFold':3,
          'adjacency_cart_depth':3,'kflod_random':True,'kflod_random_num':0,'sample_random':False,'sample_random_num':-1,'header':77,'is_feature_selection':True,'feature_selection':64,'tift':False}


rf2_ = {'my_num':20,
          'xgb1_booster':'gbtree','xgb1_eta':0.2,'xgb1_max_depth': 10,'xgb1_colsample_bytree': 0.7,'xgb1_subsample': 0.6,'xgb1_min_child_weight': 4,'xgb1_boost_round':150,
          'xgb2_booster':'gbtree','xgb2_eta':0.2,'xgb2_max_depth': 10,'xgb2_colsample_bytree': 0.7,'xgb2_subsample': 0.6,'xgb2_min_child_weight': 4,'xgb2_boost_round':150,
          'adjacency_num':10,'num_cluster':5,'cluster_times':10,'target_lambda':-5,'targets_num':8,'is_Agg_clustering':False,'Agg_clustering_num':-1,'paper_kFold':5,'inner_kFold':3,
          'adjacency_cart_depth':3,'kflod_random':True,'kflod_random_num':0,'sample_random':False,'sample_random_num':-1,'header':589,'is_feature_selection':True,'feature_selection':576,'tift':False}

all_config = {'andro':andro_,'atp1d':atp1d_,'atp7d':atp7d_,'edm':edm_,'enb':enb_,'jura':jura_,'oes97':oes97_,'osales':oscales_,'scpf':scpf_,'sf1':sf1_,'sf2':sf2_,'slump':slump_,'wq':wq_
              ,'scm1d':scm1d_,'scm20d':scm20d_,'rf1':rf1_,'rf2':rf2_}
