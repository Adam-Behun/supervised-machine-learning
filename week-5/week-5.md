# 1. Mention at least 3 concepts that you learned this week

## Decision trees and data storytelling

This was the example we went through using the data from Titanic competition. It proved that using decision trees and random forests which are collection of decision trees that help us solve the problem of collinearity is a great way to start our data analysis. Decision tress can show us which variables are impacting the variable of interest the most - they enable us to study the relationship after which we get a better idea of what the data represent and how they collerate. 


## How to create useful visualizations efficiently

The problem with visualizations is that they cannot take in as many variables as machine learning models simply because human eyes would not understand the visual. That means that we have to keep the number of variables at 2 or 3. However, our dataset might have many more columns so we need to identify the ones worth plotting. We can do this using decision trees or random forests or other algorithms but another advantage of trees and forests is that they understand numerical as well as categorical variables. This makes these two algorithms very useful is deciding which variables to include in our visualization.


## Keep categorical variables categorical

This might have been one of my mistakes at the Titanic competition where I converted categorical variables into numerical because I did not know that it might have a negative impact on my model's accuracy. I have learned that by converting categorical values into numerical I am imposing an order in that column and that is not the meaning of the actual data. There is no order in Passenger Class - it is a category. Some variables cannot be converted into numerical variables because they are a quality, a category and we cannot quantify them.


# 2. How will this week’s topic help you make better business decisions? Give one example.

This week really helped me realize the advantages of decision trees and random forests. Now, I know how to start my data analysis project or at least I know which algorithms to use to get an idea of the underlying rules and interactions between the variables. 