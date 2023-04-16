# Support Vector Machine

Support Vector Machine is looking for a hyperplane which best separates the two classes in the input data. The hyperplane is defined as the decision boundary that maximizes the margin between the two classes. The margin is the distance between the hyperplane and the closest data points from each class. Support vector machine algorithm has some advantages against other supervised learning techniques, such as it is suitable for smaller datasets as the algorithm tries to find a clear separating boundary between classes and with smaller datasets it is easier to find that boundary. It is effective in high-dimensional spaces -- able to cope with many features using kernel functions and deliver relativelly accurate, fast, and reliable results. 

# Kernel functions with SVM

This algorithm uses kernel functions which help transform data into a higher-dimensional space where it may be easier to separate the data using a hyper-plane. Kernel functions make it possible to solve non-linear problems. It is important to try mulitple kernel functions and experiment with them. There are different kernel functions used with SVMs and the most common are: linear, polynomial, radial basis function, and sigmoid. All these kernels are suitable for different datasets.
 
# Naive Bayes

It can be used in situations where speed is more important than accuracy, such as user facing applications or places where computing resources are restricted as this algorithm is, generally-speaking, faster than other supervised learning techniques. That is why it is suitable for recommendations systems ot spam filtering. Naive Bayes is mostly used for classification problems and it is based on probabilities. We should be familiar with The Bayes Theorem which uses conditional probability to update the probability of an event as new evidence or information becomes available. Seeing evidence restricts the space of possibilities. 

# Supervised Learning Algorithms Recap:
- regressions, classifications, or both

- Linear Regression
- Logistic Regression
- Linear Discriminant Analysis
- Decision Tree
- Random Forest
- k-Nearest Neighbor
- Support Vector Machine
- Naive Bayes
- Neural Networks

# ML Basics:
- We want to make predictions, lots of automatic predictions about something
  - We know what success looks like, we understand what we want
  - We have many examples of what success looks like
  - This is used mostly where it is easier to give examples of inputs and outputs.
  - Rather than giving the function we supply inputs and outputs and let the machine find the function
    - For that, the data we have is key, the algorithms already exist and work, we just have to fit the data and understand the task

# Business decisions

- Computational efficiency vs accuracy of prediction and where it is better to choose speed, such as customer facing applications where real-time data is being computed and the customer expects results immediatelly. 

- Bayes theorem and its use in real-life - update my prior beliefs based on new evidence and data. Incorporate probabilities of different outcomes into my prediction. 