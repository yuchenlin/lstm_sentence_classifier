# LSTM-based Models for Sentence Classification in PyTorch
This repo is aiming for reproducing the sentence classifcation experiments in Mou et al. (EMNLP 2016).
https://aclweb.org/anthology/D16-1046

## Datasets

### 1. IMDB Sentiment Classification
    https://drive.google.com/file/d/0B8yp1gOBCztyN0JaMDVoeXhHWm8/
    Train+dev = 600,000 (imdb.neg+imdb.pos)  550,000 for train and 50,000 for dev
    Test = 2,000 (rt_critics.test)

### 2. MR Sentiment Classification
    https://www.cs.cornell.edu/people/pabo/movie-review-data/
    Train+dev+test = rt-polarity.neg + rt-polarity.pos 
    all = 5331*2 = 10662 = 8500(train) + 1100(dev) + 1062(test) 

### 3. QC Question Classification (6 types)
    http://cogcomp.cs.illinois.edu/Data/QA/QC/
    Train(train_5500.label, 5452 = 4,800(train) + 652(dev)) + test(TREC_10.label, 500) 
    
## Performance 
### 1. LSTM-Softmax Classifier without MiniBatch or Pretrained Embedding
    "LSTM_sentence_classifier.py"
    Remark: 
    This model is the simplest version of LSTM-Softmax Classifier. 
    It doesn't use mini-batch or pretrained word embedding. 
    Note that there is not fixed lenght of the sentences.
    Its performance with Adam(lr = 1e-3) is 76.1 in terms of accuracy on MR dataset. 
    It is slower with Adam than with SGD, but the performance is much better. 
    However, it's unreasonable to use such code to train a very large dataset like IMDB.
    




