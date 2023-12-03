import networkx as nx

from datetime import date, timedelta

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import os

from functools import reduce
from pandemic_tgnn.code.preprocess import generate_graphs_britain, generate_graphs_by_day


step = 5
start_exp = 15
window = 7


os.chdir("/Italy")
labels = pd.read_csv("italy_labels.csv")
labels.loc[labels["name"]=="reggio_di_calabria","name"] = "reggio_calabria"
labels.loc[labels["name"]=="reggio_nell'emilia","name"] = "reggio_emilia"
labels.loc[labels["name"]=="bolzano","name"] = "bolzano_bozen"
labels.loc[labels["name"]=="l'aquila","name"] = "la_aquila"
del labels["id"]
labels = labels.set_index("name")

sdate = date(2020, 2, 24)
edate = date(2020, 5, 12)
delta = edate - sdate
dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
dates = [str(date) for date in dates]

Gs = generate_graphs_by_day(dates,"IT")
#labels = labels[,:]
labels = labels.loc[Gs[0].nodes(),:]

labels = labels.loc[labels.sum(1).values>10,dates]    
gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

y = list()
for i,G in enumerate(Gs):
    y.append(list())
    for node in G.nodes():
        y[i].append(labels.loc[node,dates[i]])

nodez = Gs[0].nodes()
main = pd.DataFrame(labels.loc[nodez,labels.columns[start_exp]:].mean(1))
main.columns = ["avg_cases"]
main["cases"] = pd.DataFrame(labels.loc[nodez,labels.columns[start_exp]:].sum(1))
main = main.reset_index()



os.chdir("/output")
x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
for i in range(15,79):
    try:
        x0.append(pd.read_csv("out_IT_"+str(i)+"_0.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)

    try:
        x1.append(pd.read_csv("out_IT_"+str(i)+"_1.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)

    try:
        x2.append(pd.read_csv("out_IT_"+str(i)+"_2.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)
    try:
        x3.append(pd.read_csv("out_IT_"+str(i)+"_3.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)
    try:
        x4.append(pd.read_csv("out_IT_"+str(i)+"_4.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)
        


n = x0[0]["n"]

cnt = 0
pds = []
pds_r = []
for i in range(0,len(x4)):
    tmpx = [x0[i],x1[i],x2[i],x3[i],x4[i]] # step = 5
    d = reduce(lambda p, l: p.add(l, fill_value=0), tmpx)
    del d["n"]
    d = d/step
    par = d["l"].copy()
    par[par<1]=1
    pds.append(abs(d["o"]-d["l"]))
    pds_r.append(abs(d["o"]-d["l"])/par)

pds_r = reduce(lambda p, l: p.add(l, fill_value=0), pds_r)/i
pds = reduce(lambda p, l: p.add(l, fill_value=0), pds)/i
df = pd.DataFrame({"relative":pds_r.values,"real":pds.values,"name":n })


tmp = df.merge(main,on='name')
tmp.to_csv("it_map_plot_"+str(step)+".csv")



#-------------------------------------

os.chdir("/Spain")
labels = pd.read_csv("spain_labels.csv")

labels = labels.set_index("name")

sdate = date(2020, 3, 12)
edate = date(2020, 5, 12)
#--- series of graphs and their respective dates
delta = edate - sdate
dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
dates = [str(date) for date in dates]


Gs = generate_graphs_by_day(dates,"ES")
l = Gs[0].nodes()
#l.remove("zaragoza")
labels = labels.loc[l,:]
labels = labels.loc[labels.sum(1).values>10,dates]    

#nodez = Gs[0].nodes()
main = pd.DataFrame(labels.loc[:,labels.columns[start_exp]:].mean(1))
main.columns = ["avg_cases"]
main["cases"] = pd.DataFrame(labels.loc[:,labels.columns[start_exp]:].sum(1))
main = main.reset_index()

#df = pd.DataFrame(labels.iloc[:,start_exp:].mean(1))
#df.columns = ["avg_cases"]
#df["cases"] = pd.DataFrame(labels.iloc[:,start_exp:].sum(1))
#df = df.reset_index()

os.chdir("/output")
x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
for i in range(15,62-step):
    try:
        x0.append(pd.read_csv("out_ES_"+str(i)+"_0.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)
    try:
        x1.append(pd.read_csv("out_ES_"+str(i)+"_1.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)

    try:
        x2.append(pd.read_csv("out_ES_"+str(i)+"_2.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)
    try:
        x3.append(pd.read_csv("out_ES_"+str(i)+"_3.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)
    try:
        x4.append(pd.read_csv("out_ES_"+str(i)+"_4.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)

n = x0[0]["n"]

cnt = 0
pds = []
pds_r = []
for i in range(0,len(x4)):
    tmpx = [x0[i],x1[i],x2[i],x3[i],x4[i]] # step = 5
    d = reduce(lambda p, l: p.add(l, fill_value=0), tmpx)
    del d["n"]
    d = d/step
    par = d["l"].copy()
    par[par<1]=1
    pds.append(abs(d["o"]-d["l"]))
    pds_r.append(abs(d["o"]-d["l"])/par)

pds_r = reduce(lambda p, l: p.add(l, fill_value=0), pds_r)/i
pds = reduce(lambda p, l: p.add(l, fill_value=0), pds)/i
df = pd.DataFrame({"relative":pds_r.values,"real":pds.values,"name":n })
tmp = df.merge(main,on='name')
tmp.to_csv("es_map_plot_"+str(step)+".csv")


#---------------------------------

os.chdir("/France")
labels = pd.read_csv("france_labels.csv")
#del labels["id"]
labels = labels.set_index("name")


sdate = date(2020, 3, 10)
edate = date(2020, 5, 12)

delta = edate - sdate
dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
dates = [str(date) for date in dates]

labels = labels.loc[labels.sum(1).values>10,dates]    

Gs = generate_graphs_by_day(dates,"FR")

labels = labels.loc[Gs[0].nodes(),:]

main = pd.DataFrame(labels.loc[:,labels.columns[start_exp]:].mean(1))
main.columns = ["avg_cases"]
main["cases"] = pd.DataFrame(labels.loc[:,labels.columns[start_exp]:].sum(1))
main = main.reset_index()

os.chdir("/output")
x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
for i in range(15,64-step):
    try:
        x0.append(pd.read_csv("out_FR_"+str(i)+"_0.csv"))
        
    except:
        print(i)
    try:
        x1.append(pd.read_csv("out_FR_"+str(i)+"_1.csv"))
        
    except:
        print(i)

    try:
        x2.append(pd.read_csv("out_FR_"+str(i)+"_2.csv"))
        
    except:
        print(i)
    try:
        x3.append(pd.read_csv("out_FR_"+str(i)+"_3.csv"))
        
    except:
        print(i)
    try:
        x4.append(pd.read_csv("out_FR_"+str(i)+"_4.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)

n = x0[0]["n"]

cnt = 0
pds = []
pds_r = []
for i in range(0,len(x4)):
    tmpx = [x0[i],x1[i],x2[i],x3[i],x4[i]] # step = 5
    d = reduce(lambda p, l: p.add(l, fill_value=0), tmpx)
    del d["n"]
    d = d/step
    par = d["l"].copy()
    par[par<1]=1
    pds.append(abs(d["o"]-d["l"]))
    pds_r.append(abs(d["o"]-d["l"])/par)

pds_r = reduce(lambda p, l: p.add(l, fill_value=0), pds_r)/i
pds = reduce(lambda p, l: p.add(l, fill_value=0), pds)/i
df = pd.DataFrame({"relative":pds_r.values,"real":pds.values,"name":n })


tmp = df.merge(main,on='name')
tmp.to_csv("fr_map_plot_"+str(step)+".csv")


#---------------------------------

os.chdir("/Britain")
labels = pd.read_csv("england_labels.csv")
#del labels["id"]
labels = labels.set_index("name")

sdate = date(2020, 3, 13)
edate = date(2020, 5, 12)
#Gs = generate_graphs(dates)
delta = edate - sdate
dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
dates = [str(date) for date in dates]

labels = labels.loc[labels.sum(1).values>10,dates]    

Gs = generate_graphs_britain(dates)

labels = labels.loc[Gs[0].nodes(),:]
labels = labels.loc[labels.sum(1).values>10,dates]    


#nodez = Gs[0].nodes()
main = pd.DataFrame(labels.loc[:,labels.columns[start_exp]:].mean(1))
main.columns = ["avg_cases"]
main["cases"] = pd.DataFrame(labels.loc[:,labels.columns[start_exp]:].sum(1))
main = main.reset_index()

#start_exp = 15
#df = pd.DataFrame(labels.iloc[:,start_exp:].mean(1))
#df.columns = ["avg_cases"]
#df["cases"] = pd.DataFrame(labels.iloc[:,start_exp:].sum(1))
#df = df.reset_index()

os.chdir("/output")
x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
for i in range(15,62-step):
    try:
        x0.append(pd.read_csv("out_EN_"+str(i)+"_0.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)
    try:
        x1.append(pd.read_csv("out_EN_"+str(i)+"_1.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)

    try:
        x2.append(pd.read_csv("out_EN_"+str(i)+"_2.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)
    try:
        x3.append(pd.read_csv("out_EN_"+str(i)+"_3.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)
    try:
        x4.append(pd.read_csv("out_EN_"+str(i)+"_4.csv"))
        #df.drop(df.columns[0],1))
    except:
        print(i)


n = x0[0]["n"]

cnt = 0
pds = []
pds_r = []
for i in range(0,len(x4)):
    tmpx = [x0[i],x1[i],x2[i],x3[i],x4[i]] # step = 5
    d = reduce(lambda p, l: p.add(l, fill_value=0), tmpx)
    del d["n"]
    d = d/step
    par = d["l"].copy()
    par[par<1]=1
    pds.append(abs(d["o"]-d["l"]))
    pds_r.append(abs(d["o"]-d["l"])/par)

pds_r = reduce(lambda p, l: p.add(l, fill_value=0), pds_r)/i
pds = reduce(lambda p, l: p.add(l, fill_value=0), pds)/i
df = pd.DataFrame({"relative":pds_r.values,"real":pds.values,"name":n })


tmp = df.merge(main,on='name')
tmp.to_csv("en_map_plot_"+str(step)+".csv")