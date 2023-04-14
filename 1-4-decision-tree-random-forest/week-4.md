# 1. Mention at least 3 concepts that you learned this week

# Review of where I am and what I have learned so far

Predict the future trajectory of a dependent variable using supervised learning. Thus far, using supervised learning techniques. Solving regression (numerical dependent variable) and classification (categorical dependent variable) problems. I have learned about linear regression, logistic regression (binary) ML models (both take into assumption that the data follows normal distribution). I can transform columns to have numerical or categorical values based on the model I want to use. I know that I should try multiple models before deciding which one is the best (especially with my lack of knowledge). I understand that ML is an automation in a sense that it calculates what I can calculate but is able to process many more rows enabling me to iterate algorithms faster to find the one that fits my data the most. I can use the accuracy measures for both, linear regression (r-squared) and logistic regression (confusion matrix)

Now, we are going to use decision tree, and random forests to possibly increase our chances of fitting an ML model to our dataset and predicting values with the highest probability.

# Amount of information - information entropy

Enthropy of a dataset is a measure of its randomness. The amount of information is a measure of how much reduction in uncertainty a message provides. Entropy is the measure of how much information does not provide any value - value of information is to reduce the level of uncertainty. Amount of information is measured in bits and the more uncertainty it removes, the higher the amount of information. 

# Decision tree

Decision trees are a great technique to use when one is trying to see what is happening to the data. Our objective is to find out what the function is and by having inputs and outputs we can calculate the function.
Decision trees use information entropy and amount of information to split the dataset based on which feature influences the dependent variable the most (the information that sends the most information = reduces the uncertainty the most). The algorithm continues recusively for each child node until all examples of a node belong to the same class or stopping cirterion is reached. The goal of the algorithm is to maximize the reduction of entropy (uncertainty) of the dependent variable after each split. The tree-building algorithm maximizes ifnromation gain.


# Problem of collinearity

This means that one of the independent variables have too strong of an impact on the dependent variable. Because of this we cannot determine what impact the other independent variables have. 

# Overfitting

It represent the training data well but does not generalize so it is not used representing the new dataset well. It does not understand the underlying patterns behind the training dataset and so results in inacurate predictions for the new dataset. 

# Ensemble methods

Ensemble methods are ML algorithms that help us more accurately predict the future by combining the predictions of multiple base models. 

# Random forest

Random forests help us solve the problem of collinearity because now, we do not have just one tree but a collection of trees (rnadom forest) with a large number of different predictions for each row. This prediction get averaged in numerical problems or voted for in classification problems (how many predictions predict this row to be of class a, b, or c? --> Majority wins). This way we solve the problem of collinearity. 

# Node purity

A node is considered pure if all the instances in that node belong to the same class or have the same target value. In decision tree learning, the goal is to split the data in each node in such a way that the resulting child nodes are as pure as possible, as this allows the tree to make more accurate predictions.

# 2. How will this week’s topic help you make better business decisions? Give one example.

I like that I learned about decision trees and random forests (help me understand what is going on with the dependent and independent variables) but I consider learning about information entropy and the theory behind the amount of information very useful. I think it is applicable to all areas of life, especially for me as a person trying to make useful observations from a high volume and complex datasets. Even partial knowledge of what information decreases the uncertainty in a dataset the most can have a big impact on my decisions or observations.   