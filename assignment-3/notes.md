# Engineering problem

# Identify a hand-written note
- 28*28 independent variables = 728 cells and we are checking whether a cell has ink or not
- binary
- 1 dependent variable = the one digit
- rectangular data - zeros and ones
- we have many columns - learn how to reduce the amount of variables we have
## Tools
Target could be to work with less 100 independent variable out of the original =~ 700
- Approach 1
    - Understand where the corners of the picture are - SUM the columns and delete the zeros (or even delete SUM = 300)
- Approach 2
    - From the picture identify the cells that are likely to have a contribution to the digit
- Approach 3 - you can do this after you delete the variable with no info
    - Use Principal Component Analysis  
        - Orthographic Projection
            - Still 2 dimensions but combined in the right way
            - Reduce the number of dimensions but keep the most effective 100 variables, combine them
                - Which variable is the one with most info from the dataset in the realestate.csv dataset
                    - 1. For principal component analysis, all variables must be numeric - use statsbuddy->data-preparation->one hot encoding
                    - 2. statsbuddy->dimensional reduction-> principal component analysis
                        - 3. proportion_of_variance gives me the percentage of how much info the variable has for me
                        - 4. use scatterbuddy to plot the dependent against the top 3 principal components
                        - 5. then, color it with remodel-none
                        - 6. we see that remodeling the place really matter - it is a completely different cluster

# Find the variable with the most impact on my dependent variable
- principal component analysis
- apply the analysis on both training, testing dataset and use the principal component variables to plot and use prediction
- the component variables use math to get new variables out of the ones we have