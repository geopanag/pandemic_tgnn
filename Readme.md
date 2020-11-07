# Transfer Graph Neural Networks for Pandemic Forecasting


## Data


### Labels

We gather the ground truth for number of confiremed cases per region through open data for [[Italy]](https://github.com/pcm-dpc/COVID-19/blob/master/dati-province/dpc-covid19-ita-province.csv),
[[England]](https://coronavirus.data.gov.uk) and [[Spain]](https://code.montera34.com:4443/numeroteca/covid19/-/blob/master/data/output/spain/covid19-provincias-spain_consolidated.csv}}).
We have preprocessed the data and the final versions are in each country's subfolder in the data folder. In order to map the english mobility regions from FB data to the regions for which we have ground truth, we utilized two more datasets added to the resepctive folder as well.


### Graphs

The graphs are formed using the movement data from facebook data for good disease prevention map [data](https://dataforgood.fb.com/docs/covid19/). More specifically we used the average number of people moving in one day from one region to ther other , from the [[Movement between Administrative Regions]](https://dataforgood.fb.com/tools/movement-range-maps/) datasets. We can not share this data as Facebook requires everyone to sign a data license agreement, but we can share a daily aggregated version in the graphs folder of each country.
These include the mobility maps between administrative regions until 12/5/2020, starting from 13/3 for England, 12/3 for Spain and 24/2 for Italy, which should be positioned inside a folder called Movement\_between\_Administrative_Regions in each country's subfolder.



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


