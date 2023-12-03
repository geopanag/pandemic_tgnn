import torch
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd

from datetime import date, timedelta

import os
    

def generate_features(graphs: list, 
                          labels: list, 
                          dates: list, 
                          window=7, 
                          scaled=False):
    """
    Generate node features
    
    Parameters:
    graphs (list): List of graphs
    labels (list): List of labels
    dates (list): List of dates
    window (int): Window size
    scaled (bool): Whether to scale the features
    
    Returns:
    features (list): List of features. Features[1] contains the features corresponding to y[1]
                    e.g. if window = 7, features[7]= day0:day6, y[7] = day7
                    if the window reaches before 0, everything is 0, so features[3] = [0,0,0,0,day0,day1,day2], y[3] = day3
    """
    features = list()
    
    labs = labels.copy()
  
    #--- one hot encoded the region
    for idx,G in enumerate(graphs):
        #  Features = population, coordinates, d past cases, one hot region        
        H = np.zeros([G.number_of_nodes(),window]) #+3+n_departments])#])#])
        me = labs.loc[:, dates[:(idx)]].mean(1)
        sd = labs.loc[:, dates[:(idx)]].std(1)+1

        ### enumarate because H[i] and labs[node] are not aligned
        for i,node in enumerate(G.nodes()):
            #---- Past cases      
            if(idx < window):# idx-1 goes before the start of the labels
                if(scaled):
                    H[i,(window-idx):(window)] = (labs.loc[node, dates[0:(idx)]] - me[node])/ sd[node]
                else:
                    H[i,(window-idx):(window)] = labs.loc[node, dates[0:(idx)]]

            elif idx >= window:
                if(scaled):
                    H[i,0:(window)] =  (labs.loc[node, dates[(idx-window):(idx)]] - me[node])/ sd[node]
                else:
                    H[i,0:(window)] = labs.loc[node, dates[(idx-window):(idx)]]
      
            
        features.append(H)
        
    return features






def generate_batches(graphs: list, 
                     features: list , 
                     y: list , 
                     idx: list , 
                     graph_window: int, 
                     shift: int, 
                     batch_size: int, 
                     device: torch.device, 
                     test_sample: int):
    """
    Generate batches for graphs for MPNN

    Parameters:
    graphs (list): List of graphs
    features (list): List of features
    y (list): List of targets
    idx (list): List of indices (trian, val, test)
    graph_window (int): Graph window size
    shift (int): Shift size
    batch_size (int): Batch size
    device (torch.device): Device to use
    test_sample (int): Test sample

    
    Returns:
    adj_lst (list): List with block adjacency matrices, where its smaller adjacency is a graph in the batch
    features_lst (list): The features are a list with length=number of batches.
                          Each feature matrix has size (window, n_nodes * batch_size), 
                          so second column has all values of the nodes in the 1st batch, then 2nd batch etc.
    y_lst (list): List of labels

    """

    N = len(idx)
    n_nodes = graphs[0].shape[0]
  
    adj_lst = list()
    features_lst = list()
    y_lst = list()

    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*graph_window*n_nodes
        step = n_nodes*graph_window

        adj_tmp = list()
        features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))

        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)

        #fill the input for each batch
        for e1,j in enumerate(range(i, min(i+batch_size, N) )):
            val = idx[j]

            # E.g. feature[10] containes the previous 7 cases of y[10]
            for e2,k in enumerate(range(val-graph_window+1,val+1)):
                
                adj_tmp.append(graphs[k-1].T)  
                # each feature has a size of n_nodes
                features_tmp[(e1*step+e2*n_nodes):(e1*step+(e2+1)*n_nodes),:] = features[k]
            
            
            if(test_sample>0):
                #--- val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                    
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                             
            else:
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
        
        adj_tmp = sp.block_diag(adj_tmp)
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device))
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

    return adj_lst, features_lst, y_lst



def generate_batches_lstm(n_nodes: int, 
                          y: list, 
                          idx: list , 
                          window: int  , 
                          shift:int , 
                          batch_size: int, 
                          device: torch.device , 
                          test_sample: int):
    """
    Generate batches for the LSTM, no graphs are needed in this case
    
    Parameters:
    n_nodes (int): Number of nodes
    y (list): List of targets
    idx (list): List of indices (trian, val, test)
    window (int): Window size
    shift (int): Shift size
    batch_size (int): Batch size
    device (torch.device): Device to use
    test_sample (int): Test sample

    Returns:
    adj_fake (list): A dummy list of empty adjacency matrices for the model's placeholders. Adj is not used in pure LSTM
    features_lst (list): The features are a list with length=number of batches.
                          Each feature matrix has size (window, n_nodes * batch_size), 
                          so second column has all values of the nodes in the 1st batch, then 2nd batch etc.
    y_lst (list): List of labels
    """
    N = len(idx)
    features_lst = list()
    y_lst = list()
    adj_fake = list()
    
    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*n_nodes*1
        
        step = n_nodes*1

        adj_tmp = list()
        features_tmp = np.zeros((window, n_nodes_batch))#
        
        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)
        
        for e1,j in enumerate(range(i, min(i+batch_size, N))):
            val = idx[j]
            
            # keep the past information from val-window until val-1
            for e2,k in enumerate(range(val-window,val)):
               
                if(k==0): 
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.zeros([n_nodes])#features#[k]
                else:
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.array(y[k])#.reshape([n_nodes,1])#

            if(test_sample>0):
                # val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                        
            else:
         
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]       
         
        adj_fake.append(0)
        
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append( torch.FloatTensor(y_tmp).to(device))
        
    return adj_fake, features_lst, y_lst




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




class AverageMeter(object):
    """
    Compute and store the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        




def read_meta_datasets(window, config, country_keys):
    """
    Read the datasets and create the features, labels and graphs

    Parameters:
    window (int): Window size
    config (dict): Configuration dictionary
    country_keys (list): List of country keys


    Returns:
    meta_labs (list): List of labels
    meta_graphs (list): List of graphs
    meta_features (list): List of features

    """
    os.chdir("../data")
    meta_labs = []
    meta_graphs = []
    meta_features = []
    meta_y = []
    print(country_keys[0])
    
    #------------------ Italy
    os.chdir(config['countries'][0])
    labels = pd.read_csv(config['country_labels'][0])
    del labels["id"]
    labels = labels.set_index("name")

    sdate = date(2020, config['country_start_month'][0], config['country_start_day'][0])
    edate = date(2020, config['country_end_month'][0], config['country_end_day'][0])
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    
    Gs = generate_graphs(dates, country_keys[0]) 

    labels = labels.loc[list(Gs[0].nodes()),:]
    labels = labels.loc[:,dates]    
     
    meta_labs.append(labels)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    meta_graphs.append(gs_adj)

    features = generate_features(Gs ,labels ,dates ,window )

    meta_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])

    meta_y.append(y)

    
    
    #------------------------- Spain
    os.chdir("../"+config['countries'][1])
    labels = pd.read_csv(config['country_labels'][1])

    labels = labels.set_index("name")

    sdate = date(2020, config['country_start_month'][1], config['country_start_day'][1])
    edate = date(2020, config['country_end_month'][1], config['country_end_day'][1])
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    
    
    
    Gs = generate_graphs(dates,country_keys[1])# 
    labels = labels.loc[list(Gs[0].nodes()),:]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10
   
    meta_labs.append(labels)

    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    meta_graphs.append(gs_adj)

    features = generate_features(Gs ,labels ,dates ,window )

    meta_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])

    meta_y.append(y)

    
    
    #---------------- Britain
    os.chdir("../"+config['countries'][2])
    labels = pd.read_csv(config['country_labels'][2])

    labels = labels.set_index("name")

    sdate = date(2020, config['country_start_month'][2], config['country_start_day'][2])
    edate = date(2020, config['country_end_month'][2], config['country_end_day'][2])

    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]

    
    Gs = generate_graphs(dates,country_keys[2])
    
    labels = labels.loc[list(Gs[0].nodes()),:]
    labels = labels.loc[:,dates]    
    
    meta_labs.append(labels)

    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]
    meta_graphs.append(gs_adj)

    features = generate_features(Gs ,labels ,dates ,window)
    meta_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    meta_y.append(y)


    
    #---------------- France
    os.chdir("../"+config['countries'][3])
    
    labels = pd.read_csv(config['country_labels'][3])

    labels = labels.set_index("name")

    sdate = date(2020, config['country_start_month'][3], config['country_start_day'][3])
    edate = date(2020, config['country_end_month'][3], config['country_end_day'][3])

    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]   
    
    Gs = generate_graphs(dates,country_keys[3])
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]
    
    meta_labs.append(labels)

    meta_graphs.append(gs_adj)

    features = generate_features(Gs ,labels ,dates ,window)

    meta_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])

    meta_y.append(y)
    
    os.chdir("../../code")

    return meta_labs, meta_graphs, meta_features, meta_y
    
    


def generate_graphs(dates,country):
    """
    Generate graphs for a country at a specific date
    
    Parameters:
    dates (list): List of dates
    country (str): Country
    
    Returns:
    Gs (list): List of networx graphs
    """
    Gs = []
    for date in dates:
        d = pd.read_csv("graphs/"+country+"_"+date+".csv",header=None)
        G = nx.DiGraph()
        nodes = set(d[0].unique()).union(set(d[1].unique()))
        G.add_nodes_from(nodes)

        for row in d.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])
        Gs.append(G)
        
    return Gs

