# Hinglish Offensive Tweet Detection

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


## Bidirectional GRU 
We pass the sequence through a bidirectional GRU model, concatenate the last hidden layers (forward and backward pass) then we pass this output through a Linear layer to get the outputs. We use Binary Cross entropy with logits loss (BCE + Sigmoid) as our loss function and use Adam optimizer as the optimizer, We obtain the following confusion matrix Without Mapping and with mapping respectively. 

<img src="https://github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/blob/master/Images/WoT/RNN/Screenshot_2019-10-11%20Google%20Colaboratory(2).png" width="400"> <img src ="https://github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/blob/master/Images/WT/RNN/Screenshot_2019-10-11%20Google%20Colaboratory(3).png" width ="380">

## CNN 
We use a convolutional neural network based model as out classifier.This model is an implementation of Yoon Kim et al.[1] We use Binary Cross entropy with logits loss (BCE + Sigmoid) as our loss function and use Adam optimizer as the optimizer, We obtain the following confusion matrix Without Mapping and with mapping respectively. 

<img src="https://github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/blob/master/Images/WoT/CNN/Screenshot_2019-10-11%20Google%20Colaboratory(3).png" width="415"> <img src ="https://github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/blob/master/Images/WT/CNN/Screenshot_2019-10-11%20Google%20Colaboratory(3).png" width ="380">

