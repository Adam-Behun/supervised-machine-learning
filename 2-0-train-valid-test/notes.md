# Ensemble methods
The general idea behind ensemble methods is pretty strightforward - instead of relying on one machine learning algorithm, we use a number of them and then we combine their predictions using some math functions (simple ensemble methods (avg for exmaple)) or we can use another ml algorithm to find patterns and make predictions on top of what was already prdicted by the base learners. The base learners find patterns in our dataset and then the meta model is feeded the data from base learners increasing our predictive accuracy (in most cases). The general idea to combine predictions of different algorithms as each one of them has different strengths and weekneases seems intuitive and can be applied to a number of problems to help us imporve our accuracy and lower overfit. 

# Simple methods
Once we receive the predictions from mutliple base learners, we can combine their predictions and have one result by using math opeartions such as max voting, avg, weighted avg(decide what weights to put on different results based on their individual accuracies)

# Advanced methods
Howver, there is  way to improve upon the simple ensemble methods and these are the advanced ensemble methods from which we learned about stacking, blending, bagging, and boosting. 

# Training, validation, and testing dataset
Here we have learned that we might want to split our dataset even further to be in 3 parts, namely these would be training, validation, and testing. The ration between training and validation is usually 60:40 or 50:50. However, another school of thought suggests that validation part of the dataset is not needed meaning that we would still have only two parts: training and testing - this is probably to be tested on our individual projects. 