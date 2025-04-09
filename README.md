# Spoken-Command-Recognition
Lightweight model for spoken command recognition. The architecture is a CRNN model which can accurately recognizes short commands from wave files.   The datasets for training  and testing the models consists of  image and csv files where the csv file must contain at least two columns "filename" and "label" where "filename" is a path to a wave file containing a single command utterance and "label" contains the transcription of the command.

**Requirements**

- pytorch >= 2.0
- torchaudio >= 2.0


The main script (train.py) takes the following parameters:

- train_csv_file: csv file containing the trainining data
- dev_csv_file: csv file containing the development (or validation) data 

Optionally, the following parameters can be specified:

- model: model to use ("crnn" or "rnn")
- audio_path: path to be prepended to the path in "filename" column in the csv files (to convert relative paths in csv to absolute paths if needed)
- out_dir: Output directory where the trained models will be saved

This model can be tested on the google command dataset: 

https://www.kaggle.com/datasets/neehakurelli/google-speech-commands/data
