Create an algorithm that can tell you which data visualization is the best for a particular dataset.

Subjectivity of the problem - what's the best visualization?

Difficult to give me the best visualization, but I can get a score to the combination of the variables that I use for the visualization. For the data analysis, there is usually one variable that we try to see.

Steps:
1. Primary variable of interest
2. Pick variables that influence it
3. Pick a random forest
4. Get the R^2 of that combination

Run different random forests on different combinations for the variables and see how they influence the dependent variable - pick the combination that reduces the uncertainty of the dataset the most. 

For visualizations, use the variables that reduce the uncertainty of the dataset the most. 

Once you have the most useful variables, check the datatypes of the independent variables. Create all the combinations that make sense based on the datatypes and then let the human decide which one is the most useful. 

Match variable with the dimension. 

Numerical and categorical variables

Input the dataset, get the best visualization ideas. 

https://www.statsbuddy.net/driverless-data-visualization.html