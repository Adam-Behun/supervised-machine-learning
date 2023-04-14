# Analytics
- Descriptive
- Diganostic
- Predictive
- Prescriptive

# What is ML and how it works in simple terms  

Moving into the predictive analytics part, what made me understand ML a little better was the explanation that a visualization is a way for people to find patterns in data (our brains are really good at that). However, if we want the computer to find patterns in the data, visualization is not the best way to go because for a computer it is all just zeros and ones. What we need to do is fit an ML model to our data based on the data specifics and the task at hand. Then, we feed the model with clean, preprocessed data (a training dataset). Once we are satisfied with the model's performance, we can move on to testing.   

# Classical Machine Learning and its three main types

NOTE: We will not be learning about neural nets and deep learning in this class as they are mostly used for unstructured data and we usually work with structured data in business metrics. 

- Supervised learning
    - Data is provided beforehand, this type of ML has time to train on a training dataset, it learns the relationships between inputs and outputs. Then, we use it for the new, unseen dataset once it is trained.
- Unsupervised learning
    - No data beforehand is provided to the ML, the goal is to learn the underlying structure of the data without being given specific output variables to predict. The model learns on its own.  
- Reinforcement learning
    - This type of ML uses rewards and penalties to navigate the ML in its environment. It is similar to playing a game where we learn by ourselves what to do and what not to do based on our results (rewarded or punished). 

# Regression vs Classification problems 

Regression problems in ML are trying to predict a numerical value while classification problems are trying to predict a categorical value. Continuous variables are numeric variables (weight, hight, and anything measured on a scale). Categorical variables are discrete variables that can take one of a limited number of variables. We can trnasform categorical values into numericalif we want to use regression.

# Process of predicting someone's weight

 
- `
    Identify factors that might impact someone's weight --> height, exercise habits, gender, age, diet, etc.
`

- `
      Collect data on these factors (my independent variables) and match them to the weight (my dependent variable). This is done by human. 
            Involves humans labeling the data, showing machine the pattern the independent variables (factors) have on the dependent variable (weight). Then, splitting the data into training and testing data sets.
`

- `
    Use supervised learning techniques for the machine to learn the relationship between the independent variables (factors impacting weight) and the dependent variable (weight). 
`

- `
    Once te relationship is identified and understood by the machine as well, apply this relationship to the testing data set and let the machine predict someone's weight (dependent variable) based on independent variables (factors impacting weight)
`

`
    Evaluate the results by comparing human predictions with the machine predictions. This gives us an idea on how accurate our model is. 
`

# Training and testing datasets for regression problems

The differences between a training and a testing dataset is that the training dataset has values for the dependent variable as well. We (or the machine) can calculate the correlation between dependent and independent variables formulating an equation which can then predict future values for the dependent variable. 

# Extra notes

### R-Squared
- Correlation coefficient gives me how close the DV is to PDV. We just convert it to a percentage --> R-squared --> how accurate my model is
- Check out confusion matrixes
- Black box ML models have the problem of trust. Complex algorithms which we do not understand. 
- Explainable AI - effort to understand the models
* https://teachablemachine.withgoogle.com/train

## 1. How to use ML in business after week 1

So far, I do not know much about ML but I would not be fooled as easily about what ML is and how it can help generate value in the world.