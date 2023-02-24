Titanic was about creating new columns, House prices is about finding columns that are most significant for predictions
- impute means, medians for missing values
- remove distractors
- which independent variables ifluence my dependent variables
- I need to find out which independent variables are distractors
- Use method of elimination
    - I can run random forest and remove the one with least score
    - Keep removing dependent variables until upside U --> lower score -> eliminate distractors (top score) --> eliminate distractors (lower score). You deleted too many columns.  
- Linear regression, decision tree, random forest algorithms


Use feature buddy to see which variables are impacting the house prices the most
- try deleting variables and see changes in score
- try hyperparameters
- download the predicted dataset


1. Check the N/A and replace it with none or something that is not a keyword in R

2. R expects column headers to be letters - put letters in front of the column names