# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def transarraytolist(idx_train):
    return idx_train.reshape((idx_train.shape[1],)).tolist()


def load_transductive_data_weibo(path):
    label=pd.read_csv(path+'/data/label.csv',header=0, encoding='utf-8',index_col=0).as_matrix()
    att_features = pd.read_csv(path+'/data/attributes.csv', header=0, encoding='utf-8',index_col=0).as_matrix()
    follow = pd.read_csv(path + '/data/follow.csv', header=0,index_col=0, encoding='utf-8').as_matrix()
    forward = pd.read_csv(path+ '/data/forward.csv', header=0,index_col=0, encoding='utf-8').as_matrix()
    at = pd.read_csv(path+'/data/at.csv', header=0, index_col=0,encoding='utf-8').as_matrix()

    truefeatures_list = [att_features, att_features, att_features]
    N = att_features.shape[0]
    rownetworks = [follow + np.eye(N), forward + np.eye(N), at + np.eye(N)]

    print('node:{},  metapath:{},  feature:{},   class:{}'.format(rownetworks[0].shape[0],len(rownetworks), att_features.shape,label.shape[1]))

    pd_train_20 = pd.read_csv(path + '/data/train_train20_test80.csv', encoding='utf-8', header=0)
    idx_train_random_20 = list(pd_train_20['id'])
    pd_test_20 = pd.read_csv(path + '/data/test_train20_test80.csv', encoding='utf-8', header=0)
    idx_test_random_20 = list(pd_test_20['id'])

    pd_train_50 = pd.read_csv(path + '/data/train_train50_test50.csv', encoding='utf-8', header=0)
    idx_train_random_50 = list(pd_train_50['id'])
    pd_test_50 = pd.read_csv(path + '/data/test_train50_test50.csv', encoding='utf-8', header=0)
    idx_test_random_50 = list(pd_test_50['id'])

    pd_train_80 = pd.read_csv(path + '/data/train_train80_test20.csv', encoding='utf-8', header=0)
    idx_train_random_80 = list(pd_train_80['id'])
    pd_test_80 = pd.read_csv(path + '/data/test_train80_test20.csv', encoding='utf-8', header=0)
    idx_test_random_80 = list(pd_test_80['id'])

    return att_features, rownetworks, truefeatures_list,label,idx_train_random_20,idx_test_random_20,idx_train_random_50,idx_test_random_50,idx_train_random_80,idx_test_random_80



def load_inductive_data_weibo(path,samplesize=2000):
    label = pd.read_csv(path + '/data/label.csv', header=0, encoding='utf-8', index_col=0).as_matrix()
    att_features = pd.read_csv(path + '/data/attributes.csv', header=0, encoding='utf-8', index_col=0).as_matrix()
    follow = pd.read_csv(path + '/data/follow.csv', header=0, index_col=0, encoding='utf-8').as_matrix()
    forward = pd.read_csv(path + '/data/forward.csv', header=0, index_col=0, encoding='utf-8').as_matrix()
    at = pd.read_csv(path + '/data/at.csv', header=0, index_col=0, encoding='utf-8').as_matrix()

    N = att_features.shape[0]
    rownetworks = [follow + np.eye(N), forward + np.eye(N), at + np.eye(N)]

    print('node:{},  metapath:{},  feature:{},  sampel:{}, class:{}'.format(rownetworks[0].shape[0],len(rownetworks),att_features.shape,samplesize,label.shape[1]))

    truefeatures_list = [att_features, att_features,att_features]
    train_featurelist = []
    for ii in truefeatures_list:
        train_featurelist.append(ii[0:samplesize])

    train_adjlist=[]
    for ii in rownetworks:
        train_adjlist.append(ii[0:samplesize,0:samplesize])

    pd_train_20 = pd.read_csv(path + '/data/train_train20_test80.csv', encoding='utf-8', header=0)
    idx_train_random_20 = list(pd_train_20['id'])
    pd_test_20 = pd.read_csv(path + '/data/test_train20_test80.csv', encoding='utf-8', header=0)
    idx_test_random_20 = list(pd_test_20['id'])

    pd_train_50 = pd.read_csv(path + '/data/train_train50_test50.csv', encoding='utf-8', header=0)
    idx_train_random_50 = list(pd_train_50['id'])
    pd_test_50 = pd.read_csv(path + '/data/test_train50_test50.csv', encoding='utf-8', header=0)
    idx_test_random_50 = list(pd_test_50['id'])

    pd_train_80 = pd.read_csv(path + '/data/train_train80_test20.csv', encoding='utf-8', header=0)
    idx_train_random_80 = list(pd_train_80['id'])
    pd_test_80 = pd.read_csv(path + '/data/test_train80_test20.csv', encoding='utf-8', header=0)
    idx_test_random_80 = list(pd_test_80['id'])

    return att_features[0:samplesize], rownetworks, truefeatures_list, label, train_adjlist, train_featurelist,idx_train_random_20,idx_test_random_20,idx_train_random_50,idx_test_random_50,idx_train_random_80,idx_test_random_80

