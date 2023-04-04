# How accurate is my model - regressions and classifications

Problem with accuracy was that it was positively biased and we fixed that with the validation dataset last week.
Problem is that the accuracy is not valid examiner of how accurate the algorithm actually is
    the problem was the situation with imbalanced classes
    how do we know whether the model is accurate

Performance measures:   

## Classification metrics
- Confusion matrix
  - Accuracy calculated from confusion matrix does not work well with imbalanced data
  - Precision - alternative way to see the performance of our model (first row in the confusion matrix)
    - Less susceptible to imbalance dataset
  - Recall or sensitivity - another way to look at performance (first column in the confusion matrix)
  - Specificity
  - F1 Score - combine precision and recall using harmonic mean (values from 0 to 1)

When to use which performance measure?
- Precision is about being precise - I might not to do all the tasks, but my precision is 100% because when I say 1, it is 1 for 100%
- Recall focuses on the correct predictions

Precision and recall have specific purposes and both are used 

Anytime working with ml, choose the correct performance measure. Is my dataset balanced or not?

F1 Score as harmomnic mean of precision and recall

## Regression metrics
- R-squared
- Mean squared error = L2 Loss
- Mean absolute error = L1 Loss