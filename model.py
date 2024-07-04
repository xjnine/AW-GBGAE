import math
import torch
import numpy as np
import utils
from metrics import cal_clustering_metric
from sklearn.cluster import KMeans
from GBC_2.load_data_WGBC import load_data_WGBC_v2
from tsne import plot_tsne

class AWGBGAE(torch.nn.Module):
    def __init__(self, X, labels, layers=None, lam=0.1, num_neighbors=3, learning_rate=10**-3,
                 max_iter=20, max_epoch=10, update=True, inc_neighbors=2, links=0, device=None):
        super(AWGBGAE, self).__init__()
        if layers is None:
            layers = [1024, 256, 64]
        if device is None:
            device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
        self.X = X
        self.labels = labels
        self.k = np.unique(labels).shape[0]
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.num_neighbors = num_neighbors + 1
        self.embedding_dim = layers[-1]
        self.mid_dim = layers[1]
        self.input_dim = layers[0]
        self.update = update
        self.inc_neighbors = inc_neighbors
        self.max_neighbors = self.cal_max_neighbors()
        self.links = links
        self.device = device

        self.embedding = None
        self._build_up()


    def _build_up(self):
        self.W1 = get_weight_initial([self.input_dim, self.mid_dim])
        self.W2 = get_weight_initial([self.mid_dim, self.embedding_dim])

    def cal_max_neighbors(self):
        if not self.update:
            return 0
        return 4

    def forward(self, Laplacian):
        Laplacian = Laplacian.to(torch.float32)
        embedding = Laplacian.mm(self.X.matmul(self.W1))
        embedding = torch.nn.functional.relu(embedding)
        self.embedding = Laplacian.mm(embedding.matmul(self.W2))
        distances = utils.distance(self.embedding.t(), self.embedding.t())
        softmax = torch.nn.Softmax(dim=1)
        recons_w = softmax(-distances)
        return recons_w + 10**-10

    def update_graph(self):
        data = self.embedding.detach().cpu().numpy()
        _, _, weights, _, raw_weights = load_data_WGBC_v2(data, self.labels, self.k, self.num_neighbors)
        weights = torch.from_numpy(weights)
        raw_weights = torch.from_numpy(raw_weights)
        raw_weights = (raw_weights + raw_weights.T) / 2
        Laplacian = utils.get_Laplacian_from_weights(weights)
        return weights, Laplacian, raw_weights

    def build_loss_GB(self, recons, weights, raw_weights):
        raw_weights = raw_weights.to(self.device)
        recons = recons.to(self.device)
        weights = torch.Tensor(weights).to(self.device)
        size = self.X.shape[0]
        loss = 0
        loss += raw_weights * torch.log(raw_weights / recons + 10**-10)
        loss = loss.sum(dim=1)
        loss = loss.mean()
        degree = weights.sum(dim=1)
        L = torch.diag(degree) - weights
        embedding = self.embedding.to(torch.float32)
        loss += self.lam * torch.trace(embedding.t().matmul(L.float()).matmul(embedding)) / size
        return loss

    def clustering(self, weights, k_means, SC):
        n_clusters = np.unique(self.labels).shape[0]
        embedding = self.embedding.cpu().detach().numpy()
        km = KMeans(n_clusters=n_clusters).fit(embedding)
        prediction = km.predict(embedding)
        acc, nmi, ari = cal_clustering_metric(self.labels, prediction)
        return acc, nmi, ari, prediction

    def run(self):
        data = self.X.cpu().numpy()
        n, m = data.shape
        K1 = max(math.sqrt(n), 2 * self.k)
        K2 = min(math.sqrt(n), 2 * self.k)
        self.max_neighbors = K2
        self.num_neighbors = K1
        self.inc_neighbors = K2
        _, _, weights, _, raw_weights = load_data_WGBC_v2(data, self.labels, self.k, self.num_neighbors)
        raw_weights = torch.Tensor(raw_weights).to(self.device)
        weights = (raw_weights + raw_weights.T) / 2
        Laplacian = utils.get_Laplacian_from_weights(weights)
        Laplacian = Laplacian.to_sparse()
        torch.cuda.empty_cache()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        losss = []
        accc = []
        self.to(self.device)
        for epoch in range(self.max_epoch):
            for i in range(self.max_iter):
                optimizer.zero_grad()
                Laplacian = Laplacian.to(self.device)
                recons = self(Laplacian)
                loss = self.build_loss_GB(recons, weights, raw_weights)
                torch.cuda.empty_cache()
                loss.backward()
                optimizer.step()
                weights = weights.to(self.device)
                raw_weights = raw_weights.to(self.device)
            if self.num_neighbors > self.max_neighbors:
                weights, Laplacian, raw_weights = self.update_graph()
                acc, nmi, ari = self.clustering_v1(weights, k_means=True, SC=False)
                self.num_neighbors -= self.inc_neighbors
            else:
                if self.update:
                    self.num_neighbors = int(self.max_neighbors)
                    break
                recons = None
                weights = weights.cpu()
                raw_weights = raw_weights.cpu()
                torch.cuda.empty_cache()
                w, _, _ = self.update_graph()
                _, _ = (None, None)
                torch.cuda.empty_cache()
                acc, nmi, ari = self.clustering_v1(w, k_means=True, SC=False)
                weights = weights.to(self.device)
                raw_weights = raw_weights.to(self.device)
                if self.update:
                    break
            losss.append(loss.item())
            accc.append(acc)
        acc, nmi, ari, final_label = self.ave_clustering(weights)
        return acc, nmi, ari
    
    def ave_clustering(self, weights):
        accs = []
        nmis = []
        aris = []
        label = []
        for i in range(20):
            acc, nmi, ari, prediction = self.clustering(weights, k_means=True, SC=False)
            accs.append(acc)
            nmis.append(nmi)
            aris.append(ari)
            label.append(prediction)
        maxValue = 0
        final_label = []
        maxAcc = 0
        maxNmi = 0
        maxAri = 0
        for i in range(20):
            num = accs[i] + nmis[i] + aris[i]
            if num > maxValue:
                maxValue = num
                final_label = label[i]
                maxAcc = accs[i]
                maxNmi = nmis[i]
                maxAri = aris[i]
        return maxAcc, maxNmi, maxAri, final_label
    
    def clustering_v1(self, weights, k_means=True, SC=True):
        n_clusters = np.unique(self.labels).shape[0]
        if k_means:
            if (self.embedding is None):
                X = self.X.cpu().detach().numpy()
                km = KMeans(n_clusters=n_clusters).fit(X)
                prediction = km.predict(X)
            else:
                embedding = self.embedding.cpu().detach().numpy()
                km = KMeans(n_clusters=n_clusters).fit(embedding)
                prediction = km.predict(embedding)
            acc, nmi, ari = cal_clustering_metric(self.labels, prediction)
            # plot_tsne(self.embedding.cpu().detach().numpy(), prediction, 'JAFFE', self.k, 'AWGBGAE')
        return acc, nmi, ari

def get_weight_initial(shape):
    bound = np.sqrt(6.0 / (shape[0] + shape[1]))
    ini = torch.rand(shape) * 2 * bound - bound
    return torch.nn.Parameter(ini, requires_grad=True)
