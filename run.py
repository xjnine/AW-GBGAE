from model import AWGBGAE
import torch
import data_loader as loader
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from tsne import plot_tsne
dataName = 'JAFFE'
[data, labels] = loader.load_data(dataName)
plot_tsne(data, labels, dataName, len(np.unique(labels)), 'AWGBGAE')
num_clusters = np.unique(labels).shape[0]
num_neighbors = 10
learning_rate = 10**-5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
X = torch.Tensor(data).to(device)
torch.cuda.device_count()
input_dim = data.shape[1]
layers = None
layers = [input_dim, 256, 64]
accs = []
nmis = []
aris = []
nmis_2 = []
for lam in range(10):
    gae = AWGBGAE(X, labels, layers=layers, num_neighbors=num_neighbors, lam=10**-2, max_iter=100, max_epoch=10,
                    update=True, learning_rate=learning_rate, inc_neighbors=2, device=device)
    acc, nmi, ari = gae.run()
    print(acc, nmi, ari)
    accs.append(acc)
    nmis.append(nmi)
    aris.append(ari)

print(sum(accs) / 10)
print(sum(nmis) / 10)
print(sum(aris) / 10)
print(dataName)
print("num_clusters:",num_clusters)
print('shape', data.shape)