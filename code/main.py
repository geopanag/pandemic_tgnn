#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np

import torch

from neural_utils import run_neural_model

from preprocess import read_meta_datasets

from models import arima    
import json


def experiment(args, config):
    """
    Run the experiment
    
    Parameters:
    args (argparse): Arguments for the experiment
    
    Write the results in the results folder
    """    

    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    #----------------- Read the data and add them in lists, one position for each country
    country_idx = config['country_idx']
    
    meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets(args.window, config, list(country_idx.keys()) )
    
    
    for country in country_idx.keys():
        print("Country "+country)
        idx = country_idx[country]

        labels = meta_labs[idx]
        gs_adj = meta_graphs[idx]
        features = meta_features[idx]
        y = meta_y[idx]
        n_samples= len(gs_adj)
        nfeat = meta_features[0][0].shape[1]
        
        n_nodes = gs_adj[0].shape[0]
        
        if not os.path.exists('../results'):
            os.makedirs('../results')
        fw = open("../results/results_"+country+".csv","a")

        
        for args.model in ["MPNN", "MPNN_LSTM", "LSTM","ARIMA", "AVG_WINDOW", "AVG"]:
            
            #---- ARIMA predicts by default all days until ahead
            if(args.model=="ARIMA"):
                error, var = arima(args.ahead,args.start_exp,n_samples,labels)
                count = len(range(args.start_exp,n_samples-args.ahead))

                for idx,e in enumerate(error):
                    fw.write("ARIMA,"+str(idx)+",{:.5f}".format(e/(count*n_nodes))+
                             ",{:.5f}".format(np.std(var[idx]))+"\n")
                continue

			#---- Predict each day until you reach ahead
            for shift in list(range(1,args.ahead)):
                result = []
                exp = 0

                for test_sample in range(args.start_exp, n_samples-shift):
                    exp+=1
                    print(f'Testing right now at {test_sample}')

                    #--------------------- Baselines
                    if(args.model=="AVG"):
                        avg = labels.iloc[:,:test_sample-1].mean(axis=1)
                        targets_lab = labels.iloc[:,test_sample+shift]
                        error = np.sum(abs(avg - targets_lab))/n_nodes
                        print(error)
                        result.append(error)  

                    elif(args.model=="LAST_DAY"):
                        win_lab = labels.iloc[:,test_sample-1]
                        targets_lab = labels.iloc[:,test_sample+shift]
                        error = np.sum(abs(win_lab - targets_lab))/n_nodes
                        if(not np.isnan(error)):
                            result.append(error)
                        else:
                            exp-=1

                    elif(args.model=="AVG_WINDOW"):
                        win_lab = labels.iloc[:,(test_sample-args.window):test_sample]
                        targets_lab = labels.iloc[:,test_sample+shift]
                        error = np.sum(abs(win_lab.mean(1) - targets_lab))/n_nodes
                        if(not np.isnan(error)):
                            result.append(error)
                        else:
                            exp-=1
                    else:
                        #----------------- Test neural models
                        idx_train = list(range(args.window-1, test_sample-args.sep))
                        
                        idx_val = list(range(test_sample-args.sep, test_sample, 2)) 
                                        
                        idx_train = idx_train+list(range(test_sample-args.sep+1, test_sample, 2))

                        error = run_neural_model(args.model, n_nodes, args.early_stop,
                                                idx_train, args.window, shift,  
                                                args.batch_size, y, device, test_sample,
                                                args.graph_window, args.recur, gs_adj, 
                                                features, idx_val, args.hidden, args.dropout, 
                                                args.lr, nfeat, args.epochs)
                        result.append(error)

                print("{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+",{:.5f}".format(  np.sum(labels.iloc[:,args.start_exp:test_sample].mean(1))))

                fw.write(str(args.model)+","+str(shift)+",{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+"\n")

    fw.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7,
                        help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7,
                        help='Size of window for graphs in MPNN LSTM.')
    parser.add_argument('--recur',  default=False,
                        help='True or False.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=14,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10,
                        help='Seperator for validation and train set.')
    
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    args = parser.parse_args()
    experiment(args, config)
