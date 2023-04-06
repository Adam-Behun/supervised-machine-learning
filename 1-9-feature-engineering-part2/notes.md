# Feature engineering
Work with the set of features that have the highest predictive value. To achieve this, use feature selection, construction, and extraction methods.
# Dimensionality of datasets
How many attributes/variables, columns we have in the dataset, impacts our analysis. 
NOTE*
Remeber the amount of information and enthropy lesson. Variables that reduce the unpredictability of a dataset have the highest amount of information. Variables that explain the dataset the most are the ones with highest amount of information.
## Curse of dimensionality
- having too many dimensions in a dataset can lead to the following problems
  - overfitting - model performs well on training but not good enough on testing dataset
  - decrease in model performance because it is difficult to identify the patterns in the data
  - increase in computational complexity meaning it is going to consume lots of resources to run this model
## Its solution = to solve the problem of high-dimensionality we have the following options:
- ### Feature Engineering: 
    - #### Feature Selection techniques
      - Select a subset of the variables that have the highest predictive value for the model.  
    - #### Feature Extraction techniques
      - Linear transformations
        - PCA and LDA
      - Non-linear transformations
        - t-SNE
        - UMAP
    - #### Feature Construction tecnhiques
      - Transformation
      - Imputation
        - Numeric variables
          - Mean
          - Median
          - Hot-deck
          - Regression
          - Binning
        - Categorical variables
          - Mode
          - Randomized
          - Hot-deck
          - K-Nearest neighbor
          - Binning
        - Encoding
          - label or ordinal 
          - one hot
          - dummy
          - binary
      - Deletion
      - Outliers
        - Scaling
        - Normalization and Standardization