# Hinglish-Offensive-Tweet-Detection

Keep all the data and codes in the same directory before executing.

Order of executing -
1. data_cleaner.py
2. main.py


| Models        |Without Mapping          | with Mapping  |
| ------------- |:-------------:| -----:|
| Bidirectional GRU      | 80.52 | **82.10** |
| CNN      | 83.47      |   **84.0** |
| RNN+GRU Ensemble  | **84.31**      |    83.57 |

*LSTM model was also tested, but wasn't included in the results* 

Pre-requisites - 
1. Pytorch 1.0 + 
2. numpy 
3. pandas 
4. BeautifulSoup
5. nltk 
6. gensim 
7. sklearn 
8. tqdm 
9. seaborn 
10. matplotlib
11. collections 
