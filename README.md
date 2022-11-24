# Fake News Detection

This is a comparison between LSTM and [ChordMixer](https://github.com/RuslanKhalitov/ChordMixer) on classifying fake news articles using the _Fake vs. Real News_ dataset  from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). 

## Results (test)

|                | **ROC-AUC** | **Accuracy** | **Loss** |
|:--------------:|:-----------:|:------------:|:--------:|
| **ChordMixer** |   0.9984    |    0.9984    |  0.0284  |
| **LSTM**       |   0.9995    |    0.9996    |  0.0066  |
