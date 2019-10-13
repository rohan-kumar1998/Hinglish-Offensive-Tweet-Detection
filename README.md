# Hinglish Offensive Tweet Detection

Keep all the data and codes in the same directory before executing.

Order of executing -
1. data_cleaner.py
2. main.py

## Results

| Models        |Without Mapping          | With Mapping  |
| ------------- |:-------------:| -----:|
| Bidirectional GRU      | 80.52 | **82.10** |
| CNN      | 83.47      |   **84.00** |
| RNN+CNN Ensemble  | **84.31**      |    83.57 |

*LSTM model was also tested, but wasn't included in the results* 

## Dataset 
The dataset consisted of two csv files, "HOT_Dataset_modified.csv" which contained the tweet and it's classification as one of the either classes "Not Offensive" (0), "Abusive"(1) or "Hate-Inducing"(2), it contained 3183 data points which were divided into 60% train, 10% validation, and 30% test data. The other "Hinglish_Profanity_List.csv" which contained hindi profanities and their english translation. 

## Mapping 
We consider 2 cases of mapping (onto a vector space), 
1. We map the hindi profanities and their translation to two different vectors, and then train the models. 
2. We map the hindi profanities and their translation to the same vector, and the train the same models. 

We used GloVe vector space, without freezeing the weights, and assigned random weights to the words which weren't in the GloVe vocabulary.

## Models  
### Bidirectional GRU 
We pass the sequence through a bidirectional GRU model, concatenate the last hidden layers (forward and backward pass) then we pass this output through a Linear layer to get the outputs. We use Binary Cross entropy with logits loss (BCE + Sigmoid) as our loss function and use Adam optimizer as the optimizer, We obtain the following confusion matrix Without Mapping and with mapping respectively. 

<img src="https://github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/blob/master/Images/WoT/RNN/Screenshot_2019-10-11%20Google%20Colaboratory(2).png" width="400"> <img src ="https://github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/blob/master/Images/WT/RNN/Screenshot_2019-10-11%20Google%20Colaboratory(3).png" width ="380">

### CNN 
We use a convolutional neural network based model as out classifier.This model is an implementation of Yoon Kim et al.[1] We use Binary Cross entropy with logits loss (BCE + Sigmoid) as our loss function and use Adam optimizer as the optimizer, We obtain the following confusion matrix Without Mapping and with mapping respectively. 

<img src="https://github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/blob/master/Images/WoT/CNN/Screenshot_2019-10-11%20Google%20Colaboratory(3).png" width="415"> <img src ="https://github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/blob/master/Images/WT/CNN/Screenshot_2019-10-11%20Google%20Colaboratory(3).png" width ="380">

### RNN+CNN Ensemble 
We take concatenate the pre-final layers of the above two models, and pass it through a Linear, Maxpool and Linear (in that order) to get the outputs. We use Binary Cross entropy with logits loss (BCE + Sigmoid) as our loss function and use Adam optimizer as the optimizer, We obtain the following confusion matrix Without Mapping and with mapping respectively. 

<img src="https://github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/blob/master/Images/WoT/RNN_CNN/Screenshot_2019-10-11%20Google%20Colaboratory(3).png" width="400"> <img src ="https://github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/blob/master/Images/WT/RNN_CNN/Screenshot_2019-10-11%20Google%20Colaboratory(3).png" width ="390"> 
<div style="text-align:center">
<p align="center"> 
  <img src="https://github.com/rohan-kumar1998/Hinglish-Offensive-Tweet-Detection/blob/master/Images/ENSEMBLE.png"   width="400" >
 <br> 
 Fig - RNN+CNN architecture 
</p>
</div>

## References 
1. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1746–1751,October 25-29, 2014, Doha, Qatar.c©2014 Association for Computational LinguisticsConvolutional Neural Networks for Sentence Classification
2. Proceedings of the Second Workshop on Abusive Language Online (ALW2), pages 138–148Brussels, Belgium, October 31, 2018.c©2018 Association for Computational Linguistics138Did you offend me?Classification of Offensive Tweets in Hinglish Language                                                                                                                                
                                                                                                                                
                                                                                                                                
                                                                                                                                

