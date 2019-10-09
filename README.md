# Task 
  
## Setup 

Download kaggle data from [here](https://drive.google.com/open?id=1o5ZTT1O173Qr8Ye_2Rx7f_9W37Mhdfq3) and place in data folder

1) In a conda environment or otherwise: `pip install -r requirements.txt`

2) To run on test set: `python main.py` 

## Results 
Final results on validation set can be found [here](http://bit.ly/kaggle_survey_classifier)

Overall Precision = 81%

Overall Recall = 60%

Overall f1 = 69%

Sample size error = sqrt(1/185) = 7%

NB. I have a shuffle function in the model class when training. This is to test the effect of redistributing the training data. Which should be within the sample size error in terms of quoted precision and recall. So one would not always replicate the results of the spreadsheet when running on the fly. But this effect is less than 5%. 

The overall results for the precision of the classifier are reasonably good given the time of execution for the task. There were a few things I would have spent more time on if time were available:

1) Some of my selections in the beginning, particularly for choice of features, were chosen based on intuition. Given the time it would take to convert these columns to numerical features and perform a more thorough statistical analysis on predictive power,  I decided against. If given more time I would have looked into all of the potential features since I may have missed something. 

2) I did not perform a comparison of different classification techniques, instead opting to choose one from the offset. Given the fact that the number of features is we used is only 3, one might think that a simpler classifier such as a BDT or SVM would perform as well as a deep learning approach that was ultimately employed (DNN). That said, the implementation is still fairly straight forward and the run time is fast.  

3) The segmentation analysis using PCA did not yield anything useful. If given more time I would explore other techniques perhaps using dendrograms on the full feature set, or even performing 1d visualizations and segmenting that way. I was also hoping that I could condense the feature space into two dimensions which could then be fed into the classifier. This wasn't such a useful approach since we started with a low number of features in the first place. This could be useful if performed on all 129 questions since the dimensionality there is much larger.

4) The binning of the target variable for the classifier could have been motivated better. I chose 4 classes from pre-testing different classes. If given more time I would have visualized different choices with a confusion matrix and grouped according to classes that were mistagged. This would have given more motivation for the choice.  

5) There is some bias in the analysis because I did not define a validation set (alongside training and testing). Since I tuned the probability threshold of the classifier on the test set, there is some bias in the results. I chose to do this because after rebalancing all the classes, I was not left with much data (185/class in testing, 850/class in training), and I wanted a good enough precision on the test set given the estimate on the sample size error 7%. 
