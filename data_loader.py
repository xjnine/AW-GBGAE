import scipy.io as scio
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(name):
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.array(labels).flatten()
    X = data['X']
    X = X.astype(np.float32)
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    return X, labels
