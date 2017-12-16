# CS221: Emergency Call Analysis on San Francisco Fire Department datase

## Downloading raw data and generate datasets

This is optional step, since processed data are already part of source tree

1. Run `download_datasets.py` from root folder. 
2. Then run `make_ds/sf-fire-build-ds.py`

## Vanilla RNN

Run `vanilla_rnn/rnn_predict_week.py`. This will just do "creative" prediction for one week 
on test data. To train again constants at the top of the scripts needs to be changed:
'rebuild_artifacts' set to True.

## LSTM RNN

Run `lstm_rnn/sf-fire-lstm.py` to execute RNN and LSTM time series prediction. 
The script runs training, performs test error validation and draws target vs prediction graph. 
Optional arguments:

 -n - number of neurons (default=100)
 
 -l - number of layers (deafult=3)
 
 -m - type of network, rnn or lstm (default rnn)
 
 ## Clustering
 
 Run clustering by running `clustering/location_clustering.py`. 
 
 Optional parameter '-n' - number of clusters (default = 10)

Example: 
python location_clustering.py -n 15

## Classification

For classification, just go to 
classification folder and run `classifier/classification.py`  
The ML model will be developed based on small portion of data

  

