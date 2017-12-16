# CS221: Emergency Call Analysis on San Francisco Fire Department datase

## Downloading raw data and generate datasets

This is optional step, since processed data are already part of source tree

1. Run download_datasets.py from root folder. 
2. Then run ds/sf-fire-build-ds.py

## Vanilla RNN

Run vanilla_rnn/rnn_predict_week.py. This will just do "creative" prediction for one week 
on test data. To train again constants at the top of the scripts needs to be changed:
'rebuild_artifacts' set to True.

  
