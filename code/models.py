import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd


            
def arima(ahead: int,
          start_exp: int,
          n_samples: int ,
          labels: pd.DataFrame):
    """
    Train the model
    
    Parameters:
    ahead (int): Number of days ahead to predict
    start_exp (int): Starting day to predict
    n_samples (int): Total number of samples
    labels (pd.DataFrame): Labels matrix
    max_iter (int): Maximum number of iterations
    arima_method (str): Optimization method
    arima_tol (float): Tolerance
    
    Returns:
    total_error (torch.Tensor): Total error of the model
    for_variances (torch.Tensor): List of error in each sample to compute the variance of the model
    """

    for_variances = []
    for idx in range(ahead):
        for_variances.append([])

    total_error= np.zeros(ahead)
    count = 0
    for test_sample in range(start_exp,n_samples-ahead):#
        print(f'Arima test sample {test_sample}')
        count+=1
        sample_error = np.zeros(ahead)
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            if(sum(ds.iloc[:,1])==0):
                yhat = [0]*(ahead)
            else:
                fit2 = ARIMA(ds.iloc[:,1].values, order =  (1, 0, 0)).fit(method="statespace")
                yhat = abs(fit2.predict(start = test_sample , end = (test_sample+ahead-1) ))
            
            y_me = labels.iloc[j,test_sample:test_sample+ahead]
            
            e =  abs(yhat - y_me.values)
            sample_error += e
            total_error += e

        for idx in range(ahead):
            for_variances[idx].append(sample_error[idx])

    return total_error, for_variances

            


class MPNN_LSTM(nn.Module):
    def __init__(self, nfeat: int , 
                 nhid: int , 
                 nout: int , 
                 n_nodes: int, 
                 window: int, 
                 dropout: float):
        """
        Parameters:
        nfeat (int): Number of features
        nhid (int): Hidden size
        nout (int): Number of output features
        n_nodes (int): Number of nodes
        window (int): Window size
        dropout (float): Dropout rate
        
        Returns:
        x (torch.Tensor): Output of the model
        """
        super(MPNN_LSTM, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        self.nhid = nhid
        self.nfeat = nfeat
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.rnn1 = nn.LSTM(2*nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)
        
        self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc2 = nn.Linear( nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, adj, x):
        lst = list()
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()

        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)
       
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)
        
        x = self.relu(self.conv1(x, adj, edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj, edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = torch.cat(lst, dim=1)
        
        # reshape to (seq_len, batch_size , hidden) to fit the lstm
        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))
        
        x, (hn1, cn1) = self.rnn1(x)
        
        out2, (hn2,  cn2) = self.rnn2(x)
        
        # use the hidden states of both rnns 
        x = torch.cat([hn1[0,:,:],hn2[0,:,:]], dim=1)
        skip = skip.reshape(skip.size(0),-1)
                
        x = torch.cat([x,skip], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
        
        return x
 



class MPNN(nn.Module):
    def __init__(self, 
                 nfeat: int , 
                 nhid: int, 
                 nout: int, 
                 dropout: float):
        """
        Parameters:
        nfeat (int): Number of features
        nhid (int): Hidden size
        nout (int): Number of output features
        dropout (float): Dropout rate
        
        Returns:
        x (torch.Tensor): Output of the model
        """
        super(MPNN, self).__init__()
        self.nhid = nhid
        
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid) 
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.fc1 = nn.Linear(nfeat+2*nhid, nhid )
        self.fc2 = nn.Linear(nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, adj, x):
        lst = list()
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
       
        lst.append(x)
        
        x = self.relu(self.conv1(x,adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = torch.cat(lst, dim=1)
                                   
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x)).squeeze() 
        
        x = x.view(-1)
        
        return x

    
    
    
class LSTM(nn.Module):
    def __init__(self, 
                 nfeat: int, 
                 nhid: int , 
                 n_nodes: int, 
                 window: int , 
                 batch_size: int, 
                 recur):
        super().__init__()
        """
        Parameters:
        nfeat (int): Number of features
        nhid (int): Hidden size
        n_nodes (int): Number of nodes
        window (int): Window size
        batch_size (int): Batch size
        recur (bool): Whether to use recurrent layers
        
        Returns:
        predictions (torch.Tensor): Output of the model
        """
        self.nhid = nhid
        self.n_nodes = n_nodes
        self.nout = n_nodes
        self.window = window
        self.nb_layers= 2
        
        self.nfeat = nfeat 
        self.recur = recur
        self.batch_size = batch_size
        self.lstm = nn.LSTM(nfeat, self.nhid, num_layers=self.nb_layers)
    
        self.linear = nn.Linear(nhid, self.nout)
        self.cell = ( nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).cuda()),requires_grad=True))
      
        
    def forward(self, adj, features):
        # translate into (seq_len, batch_size , n_nodes*nfeat) to fit the lstm
        features = features.view(self.window,-1, self.n_nodes)
        
        #------------------
        if(self.recur):
            try:
                lstm_out, (hc,self.cell) = self.lstm(features,(torch.zeros(self.nb_layers,self.batch_size,self.nhid).cuda(),self.cell)) 
                
            except:
                hc = torch.zeros(self.nb_layers,features.shape[1],self.nhid).cuda()                 
                cn = self.cell[:,0:features.shape[1],:].contiguous().view(2,features.shape[1],self.nhid)
                lstm_out, (hc,cn) = self.lstm(features,(hc,cn)) 
        else:
        #------------------
            lstm_out, (hc,cn) = self.lstm(features)
            
        predictions = self.linear(lstm_out)
        
        return predictions[-1].view(-1)
