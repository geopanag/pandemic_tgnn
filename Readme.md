# [Transfer Graph Neural Networks for Pandemic Forecasting](https://arxiv.org/abs/2009.08388)


## Data


### Labels

We gather the ground truth for number of confiremed cases per region through open data for [Italy](https://github.com/pcm-dpc/COVID-19/blob/master/dati-province/dpc-covid19-ita-province.csv),
[England](https://coronavirus.data.gov.uk), [France](https://www.data.gouv.fr/en/datasets/donnees-relatives-aux-tests-de-depistage-de-covid-19-realises-en-laboratoire-de-ville/) and [Spain](https://code.montera34.com:4443/numeroteca/covid19/-/blob/master/data/output/spain/covid19-provincias-spain_consolidated.csv}}).
We have preprocessed the data and the final versions are in each country's subfolder in the data folder.


### Graphs

The graphs are formed using the movement data from facebook Data For Good disease prevention [maps](https://dataforgood.fb.com/docs/covid19/). More specifically, we used the total number of people moving daily from one region to another, using the [Movement between Administrative Regions](https://dataforgood.fb.com/tools/movement-range-maps/) datasets. We can not share the initial data due to the data license agreement, but after contacting the FB Data for Good team, we reached the consensus that we can share an aggregated and diminished version which was used for our experiments. 
These can be found inside the "graphs" folder of each country.These include the mobility maps between administrative regions that we use in our experiments until 12/5/2020, starting from 13/3 for England, 12/3 for Spain, 10/3 for France and 24/2 for Italy.



## Code

### Requirements
To run this code you will need the following python packages:
 
* [numpy](https://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scipy](https://www.scipy.org/)
* [pytorch 1.5.1](https://pytorch.org/)
* [pytorch-geometric 1.5.0](https://github.com/rusty1s/pytorch_geometric)
* [networkx 1.11](https://networkx.github.io/)
* [sklearn](https://scikit-learn.org/stable/)


### Run
To run the experiments with the default settings:

```bash

cd code

python experiments.py
 
python metalearn.py
 
```

Use the script "gather_for_map.py" to aggregate data in the output folder to produce the map plots and the "tl_base.py" for the TL_BASE baseline. 


