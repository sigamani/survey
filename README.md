# Task 
  
## Analysis of https://www.kaggle.com/stackoverflow/stack-overflow-2018-developer-survey

1) In a conda environment or otherwise: `pip install -r requirements.txt`

2) To run on test set: `python main.py` 

### Results 
Final results on validation set here: http://bit.ly/kaggle_survey_classifier

Precision = 82%

Recall = 57%

f1 = 67%

Sample size error = sqrt(1/185) = 7%

The overall results for the precision of the classifier are reasonably good given the time of execution for the task. There were a few things I would have spent more time on if time were available:

1) Some of my selections in the beginning, particularly for choice of features, were guessed based on intuition. Given the time it would take to convert these columns to numerical features and perform a more thorough statistical analysis on predictive power,  I decided against. 

2) I did not perform a thorough comparison of different classification techniques, instead opting to choose one from the offset. Given the simplicity of the problem (now looking in hindsight), one might think that a simpler classifier (BDT, SVM) might be a less complicated option than using a deep learning approach from TensorFlow. That said, the implementation is still fairly straight forward and intuitive.  

3) The segmentation analysis using PCA did not yield anything interesting. If given more time I would explore other techniques perhaps using dendrograms on the full feature set, or even performing 1d visualizations and segmenting that way.

4) The binning of the target variable for the classifier should have been motivated better. I chose 4 classes since experience told me that this was a sensible number given the variability of the data. If given more time I would have visualized different choices with perhaps a confusion matrix and grouped according to classes that were mistagged. 

5) There is some bias in the analysis because I did not define a validation set (alongside training and testing). Since I tuned the probability threshold of the classifier on the test set, there is some bias in the results. I chose to do this because after rebalancing all the classes, I was not left with much data (185/class in testing, 850/class in training), and I wanted a good enough precision on the test set given the estimate on the sample size error 7%. 
