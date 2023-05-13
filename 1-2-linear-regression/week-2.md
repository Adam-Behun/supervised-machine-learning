# Linear regression ML
- In regression problems we can only use numerical variables (if we have any categorical values, we have to convert them to numerical - preprocessing). Independent variables are the ones that have influence on our dependent variable - dependent variable is the one we are trying to predict. We can have as many independent variables as we like (later topic).

- Another advantage of using ML to find patterns in data and not visualizations is that visualizations cannot take as many variables as ML models can. We can let the computer understand the impact of more than 3 variables.

- Linear regressions on a graph are straight lines which makes it easy to visualize how we can predict its future trajectory and points. 
- y = a + bx is the model for linear regression
  
- If we have a dependent variable that we are trying to predict, we need to have a training dataset which has values for the dependent variable as well and then we fit an ML model using the method of least squares onto that training dataset. After that, the model finds the relationship between the dependent variable and independent variables and creates a new column (Predicted using LR). Then it calculates the R-Squared which shows us how well the model  fits our data. We could calculate all this manually but it is not scalable or efficient --> in that way ML is an automation and helps us speed up the process of predicting future values with levels of certainty.
 
# Interpretation of results from R and statsbuddy
- .csv file with predicted using LR column - this column shows us the predicted values for the dependent variable using LR 
- R-squared value - shows how much of the variance in dependent variable is explained by the independent variables
- Significance codes - show us which independent variables impact the dependent variable the most

# Evaluating the LR model's accuracy
- The Method of Least Squares is a technique to minimize the sum of vertical offsets making sure that our data fits the linear regression line as closely as possible - it calculates the optimal coefficients for the equation. There is a minimum distance between our data points and the model's line. The method of least squares is used with linear regression problems.

- R-squared in linear regression models is a measure used to check how well the model fits the data. It ranges from 0 to 1. A 12% R-squared score means that our model explains 12% of the variability in our dependent variable. 

# EXTRA:
1. The method of least squares is a technique used to find the values of the coefficients (b0, b1, b2, ..., bn) that minimize the sum of the squared differences between the predicted values (y) and the actual values (y') of the dependent variable.
2. Sign up for The Kaggle Titanic Competition
3. Difference between training and testing data. In training data, there is DV provided. In testing data, we are trying to predict the DV of the testing dataset. 
4. ML model describes the relationship between the variables - it is a function 
5. Simplify your ML model
     - Seeing whether b = 0 is usefull because I can see which variables have impact on the dependent variable and which do not
     - we can delete the variables that do not have impact on my dependent variable
     - this can help me simplify the ML model and only use the data that has impact on my predictions
     - the most simple model "is the best"
     - which variables impact my dependent variable --> p value, the lower the p value the higher the chance that b is not zero. More stars, more influence. 
# Business usecases
>If I were to organize use cases for linear regression into directories, I would likely group them based on the type of problem being solved and the type of data being used. Some possible directories I might create include:

1. Sales forecasting: This directory would include use cases where linear regression is used to predict future sales based on historical data and other relevant factors.

2. Pricing optimization: This directory would include use cases where linear regression is used to optimize pricing based on factors such as cost, demand, and competition.

3. Quality control: This directory would include use cases where linear regression is used to predict quality based on factors such as raw materials, process variables, and environmental conditions.

4. Financial analysis: This directory would include use cases where linear regression is used to analyze financial data, such as stock prices, interest rates, and economic indicators.

5. Medical research: This directory would include use cases where linear regression is used to analyze medical data, such as patient outcomes, treatment efficacy, and risk factors for disease.

6. Image processing: This directory would include use cases where linear regression is used to process image data, such as image recognition, object detection, and image segmentation.

7. Natural Language Processing: This directory would include use cases where linear regression is used to process natural language data, such as sentiment analysis, language classification, and part-of-speech tagging.

8. Predictive maintenance: This directory would include use cases where linear regression is used to predict the failure of machines and equipment based on sensor data, usage patterns, and other factors.

9. Fraud detection: This directory would include use cases where linear regression is used to detect fraudulent transactions based on patterns in financial data.

10. Environmental modeling: This directory would include use cases where linear regression is used to model environmental factors such as weather, climate, and pollution levels.
