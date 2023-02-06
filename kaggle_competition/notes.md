https://www.kaggle.com/competitions/titanic

Random submission file 
- Tells me what structure kaggle takes in - my submission needs to look like the gender_submission file

- I submitted the first gender_submission.csv file and received a score of 

# Titanic game

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


Approach 1:
Not using cabin column at all
Inserting the median of 28 to the missing values of age
Using S for embark because:
C has 270
Q has 123
S has 914

Using 14.4542 for fare missing value

Prediction.csv has a cutoff of 0.7

## Improve the prediction ideas

Get a higher score and get into the 10% of leaderboard and get a batch

1. Set of independent variables
2. Play with the cutoff
3. Try different algorithms
4. Innovative new columns
   1. Do not use the name as an independent variable -- we can extract the social status by taking the Mrs, or Ms from the name column. Only take the major ones and group the rare ones into a column called other
   2. Calculate the family size -- using the columns siblings, spouse, and parents, childs
5. Go to discussion and check what people did how they have changed the data