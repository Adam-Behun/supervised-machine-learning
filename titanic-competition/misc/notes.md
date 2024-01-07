# Approaching an ML problem
- Problem Definition
- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Selection
- Model Training
- Model Evaluation
- Model Tuning
- Model Deployment
- Model Maintenance
- Results Interpretation

# Problem definition
My job is to predict who is going to survive The Titanic Disaster and who is not, based on some features that are known about the passengers. I have received two datasets: One dataset titled train.csv and the other titled test.csv. Train.csv contains 891 passengers and their information including the“ground truth” - whether the passenger survived the accident. The test.csv dataset contains similar information but does not disclose the “ground truth” for each passenger. My task is to predict whether the 418 passengers of the test.csv survived the accident. I am going to use supervised learning machine learning models to find the patters in the train.csv dataset and after the patters reveal themselves I will apply this model to the test.csv dataset. 

# Data collection
I have started this project by importing these two datasets into my local machine. The datasets are relativelly small, simple, and easy to work with. The data is provided to me in a .csv format. 

# Exploratory Data Analysis
It was important to start this project with some basic exploratory data analysis. During this part of my work I have found out that in the train.csv:
- 891 rows and 12 columns
- missing values in the 'Age', 'Cabin', 'Embarked' columns
- 62% of all passengers died in the accident
- clear relationship between 'Pclass' and 'Survived' variables
    - passengers in higher class are more likely to survive
- 'Name' column can be used after extracting titles
- clear relationship between newly-created 'Title' variable and 'Survived'
- passengers with more letters in their 'Name' variable are more likely to survive
    - this might mean that they have complex titles - higher status?
- 65% of passengers are male
- 74% of female passengers survived while only 19% of male survived
- correlation between 'SibSp' 'Survived' and 'Parch' 'Survived' is not that strong, so we can create a 'FamilySize' which helps to find the relationship
- 72% of passengers embarked in the port 'S'
- more expensive ticket means higher survival chances

After I gained a better understanding of the train.csv dataset I can do some feature engineering work to prepare my dataset for model training.

# Feature Engineering
After I had a decent understanding of the dataset using visualizations and simple computations I moved onto the feature engineering part of my project.
- The first thing was to delete the columns that provided little to no value for my work. I decided to drop the columns Ticket, Cabin, PassengerId, and Died (was created to better visualize the relationship within sex and surival rate).
- Next, I split the 'Name' column, create a new column 'Title', and extract the titles of different passengers. This proved to be really useful and relatively simple to do. I splitted the column Name, found out that there are 17 different titles for representing various characteristics of the passengers on-board. I have concluded that there were 6 "social classes": 1. Officer, 2. Royalty, 3. Miss, 4. Mrs, 5. Mr, 6. Master and I have matched the previous titles provided (17) to these new titles (6). After I had the Title column ready, I dropped the name column which provided little value (most likely). 
- After this I decided to convert 'Sex', 'Embarked', and 'Title' columns into numerical with thinking that it is going to make my logistic regression model more accurate. 
- My first analysis showed me that there are missing values in 'Age' column. I have decided to compute a median for both, male and female 'Age' variables and then impute these values to the respecitve missing values of my train dataset. 
- I also saw that there are 2 rows with missing 'Embarked' values but because they were just two I decided to drop those rows. 
- I created a new variable called 'Fam_Size' which calculates the size of the family using 'SibSp' and 'Parch' 
 
After these adjustments were made, there are no missing values in any of my variables. New variables created are 'Fam_Size', 'Title'. Some space for improvement is in extracting some more information from the columns'Cabin', 'Ticket', which can store some information related to my dependent variable 'Survived'. 
- Another technique to improve my models accuracy is to normalize or rescale the 'Age' and 'Fare' columns for the values to fall between 0 and 1. After all these steps were done, I had a clean dataset and I was ready to fit an ML model to it. 

# Model Selection
I have tried Logistic Regression, Decision Tree, and Random Forest algorithms to solve this problem. My best official scores from kaggle.com were the following: 
- Logistic Regression =  0.787
- Decision Tree = 0.775
- Random Forest = 0.785
- I cannot conclude that Logistic Regression is the best model to use with this dataset because I have used slightly different techniques to prepare my dataset for different models. These are my notes for the different models I have tried. As I mentioned, there is predictive value in some more variables than I used, so there definitely is space for improvement. However, my current skills with feaute engineering or hyperparameter tuning are not sufficient to accomplish a better score. After I was done with my Random Forest algorithm I ran a code which showed me how big of an impact different independent variables have on my dependent variable and it seems that 'Sex' ('Male'), 'Title' ('Mr'), 'Fare', 'Age', 'Sex ('Female'), 'Pclass' ('3') were the most impactful variables in that order.  

# Results Interpretation
My best score was achieved using Logistic Regression and my prediction was accurate with an official score of 0.78708 which ranks me at 1053 out of 13953 players. In this paper, I will describe how I achieved this score and what I might try to improve my score further. The score of 78.7% means that I am 78.7% confident that my model can predict the future of who is going to survive and who is not 78.7% of the time. My accuracy score from sklearn.metrics was 84.27%.