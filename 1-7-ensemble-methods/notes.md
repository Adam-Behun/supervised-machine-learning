# Ensemble methods
I have talked about ensemble methods in ML before when we learned about random forest. Ensemble methods in general are helping us improve the accuracy of our predictions by combining the predictions of multiple models. For example, the collection of multiple decision trees is the ensemble method of random forest which tends to perform better than the individual decision tree by itself. We need to understand the tradeoff for running these more complex algorithm and that is the run-time - it tends to be higher. This week we have learned about some other ensemble methods which can be divided into two groups, bagging and boosting. 

# Bagging ensemble methods
Bagging ensemble methods esentially involve building multiple models on different subsets of the training dataset after which their predictions get combined into 1 prediction. Random forest is a bagging technique as it creates an N amount of independent decision trees which get consolidated into a singular prediction which is our final result. 

# Boosting ensemble methods
Boosting ensemble methods on the other hand work by iterations, they improve based on the results from the previous prediction. They start with 1 decision tree and after seeing its coefficient these methods make adjustments to the data as well as to the tree. From the boosting ensemble methods in machine learning we have learned about Adaptive Boosting - based on decision trees, Gradient Boosting - improved Adaboost algorithm, and Extreme Gradient Boosting - improves the speed of gradient boosting algorithm. Extreme Gradient Boosting is a poweful algorithm which improved the Gradient Boosting using parallel processing, tree pruning, caching and it also handles missing data in a slightly different manner. 

# Business impact
Ensemble methods help us to improve the accuracy of predictions especially in noisy and complex datasets which are common in businesses. The usecases for these machine learning algorithms are similar to the other supervised learning techniques but the thing to remember is that they tend to be slower as they combine predictions from multiple models which is not really ideal if we must have the result as soon as possible. 

# Supervised Learning Algorithms Recap:
- regressions, classifications, or both

- Linear Regression
- Logistic Regression
- Linear Discriminant Analysis
- Decision Tree
- Random Forest
- k-Nearest Neighbor
- Support Vector Machine
- Naive Bayes
- Neural Networks

# ML Basics:
- We want to make predictions, lots of automatic predictions about something
  - We know what success looks like, we understand what we want
  - We have many examples of what success looks like
  - This is used mostly where it is easier to give examples of inputs and outputs.
  - Rather than giving the function we supply inputs and outputs and let the machine find the function
    - For that, the data we have is key, the algorithms already exist and work, we just have to fit the data and understand the task