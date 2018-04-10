import numpy as np
import pandas as pd
import scipy.sparse





def getData():
    '''
    train, test = getData()
    '''
    train = pd.read_csv('../output/train_pre.csv', )
    test = pd.read_csv('../output/test_pre.csv')
    return train, test


def getFeature(Name, CharFlag=False):
    '''
    train_features, test_features = getFeature('', CharFlag=True)
    '''
    if CharFlag:
        train_features = scipy.sparse.load_npz('../output/train_features.npz')
        test_features = scipy.sparse.load_npz('../output/test_features.npz')
        return train_features.todense(), test_features.todense()
    else:
        if Name == 'fastEeb':
            fastEeb = np.load('../output/embedMatrix.npy', encoding='latin1')
            return fastEeb
        else:
            globeEbd = np.load('../output/embedMatrixglove.npy', encoding='latin1')
            return globeEbd


def getSequence():
    '''
    train_sequence, test_sequence = getSequence()
    '''
    train_sequence = np.load('../output/x_train_seq.npy', encoding='latin1')
    test_sequence = np.load('../output/x_test_seq.npy', encoding='latin1')
    return train_sequence, test_sequence

def getIP(train,test):
    train_ip = train[['ip_0','ip_1', 'ip_2', 'ip_3']]
    train_ip['index'] = 1
    test_ip = test[['ip_0','ip_1', 'ip_2', 'ip_3']]
    test_ip['index'] = 0
    result = pd.concat([train_ip,test_ip])

    result = pd.get_dummies(result, prefix=['ip_0','ip_1', 'ip_2', 'ip_3'],
                            columns=['ip_0','ip_1', 'ip_2', 'ip_3'],drop_first=True)
    train_ip = result[result['index']==1]
    test_ip = result[result['index']==0]
    return train_ip,test_ip