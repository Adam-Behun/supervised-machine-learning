Steps: 
1. Join the two files, join the training and testing dataset to create 1 dataset
2. Understand (scrutinize) the quality of the data - missing values in the file (age, embark, cabin columns)
   1. If there are too many missing values, might not even use them. Which columns are worth keeping as independent? -- 60% of values must be there, otherwise do not use that column as an independent variable
   2. How to treat the missing values?
      1. Use mean or median of all the values in that column
      2. Categorical columns with missing values are to be treated by inserting the most common value is to be inserted
3. Select dependent and independet variables based on your choosing after understanding the data and its impact on the dependent variable
4. Into statsbuddy, then r-cloud
5. Change the resulting file into the same format as the provided gender_submission.csv


# Final report

# Analyzing the dataset

What do the different columns in the dataset mean to get a sense of what I am working on. The goal is to predict which rows representing individual people are going to survive. 

My best score was achieved using Logistic Regression and my prediction was accurate with an official score 0.78708 which ranks me at 1053 out of 13953 players. In this paper, I will describe my how I achieved this score and what I might try to improve my score further. 

# What do the final results mean

The score of 78.7% means that I am 78.7% confident that my model can predict the future of who is going to survive and who is not 78.7% of the time. My accuracy score from sklearn.metrics was 84.27%. 

# Exploratory data analysis

