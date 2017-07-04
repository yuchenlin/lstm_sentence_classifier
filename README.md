# LSTM-based Models for Sentence Classification in PyTorch
This repo is aiming for reproducing the sentence classifcation experiments in Mou et al. (EMNLP 2016).


## Datasets:

### 1. IMDB Sentiment Classification: 
    https://drive.google.com/file/d/0B8yp1gOBCztyN0JaMDVoeXhHWm8/ 
    Train+dev = 600,000 (imdb.neg+imdb.pos)  550,000 for train and 50,000 for dev
    Test = 2,000 (rt_critics.test)

### 2. MR Sentiment Classification:
    https://www.cs.cornell.edu/people/pabo/movie-review-data/ 
    Train+dev+test = rt-polarity.neg + rt-polarity.pos 
    all = 5331*2 = 10662 = 8500(train) + 1100(dev) + 1062(test) 

### 3. QC Question Classification (6 types):
    http://cogcomp.cs.illinois.edu/Data/QA/QC/
    Train(train_5500.label, 5452 = 4,800(train) + 652(dev)) + test(TREC_10.label, 500) 
