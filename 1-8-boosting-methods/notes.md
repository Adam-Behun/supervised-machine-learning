# Boosting ensemble method
- sequential
- the trees are constantly improving, the growth is not independent, they improve on each other, it evolved based on the past
- they have decision tree at the foundation level
- start with 1 decision tree and after looking at the coefficient tree make adjustments to the data and the tree, then iterate

# Adaptive Boosting
- Adaboost is an ensemble method based on decision trees
- model 1 is created (iteration 1)
- increase the weights of missclassified rows and duplicate them in the training data
    - focus more on the missclassified rows
        - force the algorithm to play with the weights of the rows that have been missclassified
        - iterate this process, 500 times more or less to get the super-imposition of the rules

# Gradient Boosting
- improvement on Adaboost which was the first boosting algorithm
- gradient descent algorithm - math optimization algorithm
- try this with the housing-prices competition

# Extreme Gradient Boosting
- improved gradient boosting - it's faster
- using parallel processing, tree pruning, caching, handle missing data
- improved implementation, math is similar to the gradient boosting
- compromise between the speed and the accuracy

# Bias and Variance 
https://www.youtube.com/watch?v=EuBBz3bI-aA

# Ridge Regression
https://www.youtube.com/watch?v=Q81RR3yKn30

# Lasso Regression
https://www.youtube.com/watch?v=NGf0voTMlcs

# Elastic Net Regression
https://www.youtube.com/watch?v=1dKRdX9bfIo