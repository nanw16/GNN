import torch
import torch.nn as nn
from models.layers import GCN_layer, GAT_layer, APPNP_layer
from utils.preprocess import norm_adj, norm_adj2, norm_hop_adj, SparseDropout

class Model_GCN(nn.Module):
    def __init__(self, n_dims, A, dropout=0.5):
        super(Model_GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.A = norm_adj(A, True).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.layers = nn.ModuleList()
        for i in range(len(n_dims)-1):
            self.layers.append(GCN_layer(n_dims[i], n_dims[i+1]))
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, self.A)
            x = self.relu(x)
            x = self.dropout(x)
        return self.layers[-1](x, self.A)
 
class Model_GAT(nn.Module):
    def __init__(self, n_dims, n_heads, A, dropout=0.6, attn_dropout=0.6, alpha=0.2):
        super(Model_GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.s, self.t, self.I = self.get_I(A)
        self.I = self.I.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.layers = nn.ModuleList()
        n_in = n_dims[0]
        for i in range(len(n_dims)-2):
            self.layers.append(GAT_layer(n_in, n_dims[i+1], n_heads[i], attn_dropout, alpha, concat=True))
            n_in = n_dims[i+1] * n_heads[i]
        self.layers.append(GAT_layer(n_in, n_dims[-1], n_heads[-1], attn_dropout, alpha, concat=False))
    def forward(self, x):
        x = self.dropout(x)
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, self.s, self.t, self.I)
            x = self.elu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, self.s, self.t, self.I)
        return x
    def get_I(self, A):
        A = norm_adj(A, self_loop=True)
        s, t = A._indices()
        N = A.size(0)
        s1 = s.tolist()
        I = torch.sparse_coo_tensor([s1, list(range(len(s1)))], torch.ones(len(s1)), (N, len(s1)))
        return s, t, I
    
class APPNP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, A, dropout=0.5, dropout2=0.5, k=5, alpha=0.1):
        super(APPNP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.appnp = APPNP_layer(n_out, k, A, dropout2, alpha)
        self.reset_param()
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.appnp(x)
        return x
    def reset_param(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        
class DAGNN(nn.Module):
    def __init__(self, n_in_dim, n_hid_dim, n_out_dim, A, hop, dropout=0):
        super(DAGNN, self).__init__()
        self.hop = hop
        self.A = norm_adj(A, True)
        self.linear1 = nn.Linear(n_in_dim, n_hid_dim)
        self.linear2 = nn.Linear(n_hid_dim, n_out_dim)
        self.s = nn.Parameter(torch.FloatTensor(n_out_dim, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_param()
    def forward(self, x):
        A = self.A.to(x.device)
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(self.dropout(x))
        out = [x]
        for _ in range(self.hop):
            x = torch.sparse.mm(A, x)
            out.append(x)
        H = torch.stack(out, dim=1)
        S = torch.sigmoid(torch.matmul(H, self.s))
        S = S.permute(0, 2, 1)
        H = torch.matmul(S, H).squeeze()
        return H
    def reset_param(self):
        gain = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(self.s, gain=gain)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear1.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear2.weight, gain=1)