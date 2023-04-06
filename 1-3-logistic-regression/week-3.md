# 1. Mention at least 3 concepts that you learned this week

# What are classification problems
Classification problems try to predict the categorical class label of a new data point based on its independent variables. They are different from regression problems because regressions predict continous (numerical) values, classifications predict discrete (categorical) values. There are many types of classification problems, we have studied the binary classification problem where the algorithm picks the class label of a data point from 2 classes, zero and one respectively. The logic of 0 and 1 can be used for many usecases.  

If I have a categorical dependent variable I can use a classification problem to predict its future values. The independent variables can be either numeric, or categorical. Categorical variables can be transformed into numeric using one-hot encoding. We need to create classes for the dependent variable, cannot predict exact value like we did with regressions. 

# What does it mean to fit an ML model
We have learned that many problems (if not all depending on their complexity) can be transformed from numeric into categorical and so we can use either regression models or classification models based on their accuracy. We fit an ML model by trying a number of different ones in an effort to have the highest accuracy --> anything can be predicted but with low accuracy prediction is just a guess. Different algorithms have different techniques to understand their accuracy.  

For linear regression model to work we need all variables to be numeric. Regressions are trying to fit a line or a curve to the data. 

# Logistic regression
Dependent variable is binary. It models the probability that a given inpu belongs to a class (0 or 1). 
Y-axis represents the predicted probability and the x-axis represent the indepedent variables. Y-axis only goes from 0 to 1 and we can adjust the cut-off based on the model's accuracy. We use logistic function to give the results boundaries --> how does this work?
The logistic regression graph shows a sigmoidal or S-shaped curve (Age vs height - starts really slow, then jumps, then slows down), that represents the relationship between the independent variables and the dependent variable. The curve maps the predicted probability of a binary dependent variable given the values of the independent variables. 

# Confusion matrix and accuracy of logistic regression
Confusion matrix is used to analyze the accuracy of a binary classification model. It is a table providing values of true positives, false positives, true negatives, false negatives and after that we can calculate (r calculates) the accuracy of our model.  

# One-hot encoding
If we have a categorical value color with values [red, green, blue] and we want to use a classification model, we can create 3 columns [color_red, color_green, color_blue] and use 0 and 1 to indicate the value of a point (row). This way the variable is transofrmed from categorical into numeric.  

# Missing values
Data imputation process. Missing values impact ML model's accuracy and need to be corrected before training the model. There are a number of different strategies to work with missing values
- mean, median, mode encoding (fill in the gaps with those values)
- more advanced methods

# Numeric and Categoric (continous and discrete)
Binary datasets are datasets where values of a column can take only one of two values (0 or 1). They are used on binary classification problems to predict the class label based on a point's features.
Variables can be either numeric (continous) or categorical (discrete)
 
# 2. How will this week’s topic help you make better business decisions? Give one example. 
I can use different algorithm to predict the value of my dependent variable. I can also now convert regression problems to classification problems and see which is performing better.

# Extra notes
Can I use linear regression to find out which variables impact the dependent variable the most and then use logistic regression to label the dependent variable into a class. 

Accuracies of regressions and classifications models.
What can I use to solve an ML problem
- regressions
- classifications
- different independent variables selected
- change cut-off in classification problems
- hyperparameter tuning later

Training data has values for all variables --> it is training the dataset --> this is supervised learning --> data must be provided beforehand. 

Pre-determined parameters in ml models are params that are set before training the model.

In the Titanic game, a person is 1 or 0 and that means that I have a 65% confidence in my resulting dataset --> it does not go to one row but all together I have a 65% confidence in my results.