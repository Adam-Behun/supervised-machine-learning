# Tasks
- Check for the highest correlation with the target variable:

Train data:
Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition,SalePrice

Test data:
Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition

Highest correlation with the target variable:

- "OverallQual", 
- "GrLivArea", 
- "GarageCars", 
- "GarageArea", 
- "TotalBsmtSF", 
- "1stFlrSF", 
- "FullBath", 
- "YearBuilt"

# Problem definition
This competition is based around the problem of deciding what impacts the prices of houses in a certain area and then predicting a price for a specific house based on the features we have. The dataset we are provided includes information about different houses in Ames, Iowa and using this data and our machine learning skills we are to predict the final price of each house. 

# EDA - Exploratory Data Analysis
My exploratory data analysis showed that in the training dataset there are 1460 observations while in the testing dataset there is 1459 observations. Both datasets contain quantitative and qualitative variables. In the training dataset there is 19 attributes with missing values. My target variable is SalePrice and it is a continuos variable, however, it does not follow normal distribution which means that some transformation might be needed to imrpove the model's performance. Another important fact my EDA showed is that the following variables: "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF", "FullBath", and "YearBuilt" have the highest correlation with the target variable "Saleprice" and so have been selected to build my machine learning model and have the highest chance to perform well using some supervised machine learning techniques I will talk about later.

# Feature engineering
In my code I have removed outliers from "GrLivArea" column thinking that removing the values where "GrLivArea" is smaller than 4500 represents the dataset better. I believe these values might have been in the dataset due to some error in the data collection or maybe they were correct, anyway, I do not believe they have represented my dataset correctly and this would negativelly impact my prediction accuracy. Moreover, I have transformed the target variable "SalePrice" using log transformation which is used for regression problems when the variable has a long-tailed distribution. This is making the target variable more normally distributed assuming that it follows normal distribution. 

## Missing values
For the missing values in "Functional", "Electrical", "KitchenQual", "PoolQC", "Exterior1st", "Exterior2nd", and "SaleType" variables I have used mode (most common value) for imputation. For the missing values in "GarageYrBlt", "GarageArea", and "GarageCars" I have imputed missing values with 0. For the missing values in "GarageType", "GarageFinish", "GarageQual","GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", and "BsmtFinType2" I have replaced them with "None". 

## Dropping columns
I have dropped the variables "Utilities", "Street", and "PoolQC". 

## Creating new features
The first feature called "Total_sqr_footage", was calculated as the sum of the square footage in the basement finished area ("BsmtFinSF1" and "BsmtFinSF2"), "1stFlrSF", and "2ndFlrSF". The second feature "Total_Bathrooms", was calculated as the sum of the number of "FullBath", plus "HalfBath" divided by 2, plus the number of "BsmtFullBath" plus "BsmtHalfBath" divided by 2. The third feature, "Total_porch_sf", was calculated as the sum of the square footage of various types of porches and decks including "OpenPorchSF", "3SsnPorch", "EnclosedPorch", "ScreenPorch", and "WoodDeckSF". 

# Model Selection
To select the best model for the given dataset, I have trained various regression models, including Linear Regression, Ridge Regression, Lasso Regression, Elastic Net Regression, Random Forest, Gradient Boosting, and Extreme Gradient Boosting. However, I have learned that fitting a model to the dataset is complex task. Givem that, I decided to combine multiple regression models to achieve better performance. To achieve this, I used a technique called blending, which involves training different models on the same dataset and then combining their predictions to calculate a final prediction. By doing this, blending allowed me to emphasize the strengths of various models, each of which could have captured different information about the dataset. To implement blending, I selected Lasso Regression, Ridge Regression, Elastic Net Regression, Gradient Boosting, and Extreme Gradient Boosting as my base models, as they performed well in previous experiments. To combine their predictions, I used Stacking, a meta-learning method that trains a model to learn how to combine the outputs of the base models. To avoid overfitting, I used regularization techniques such as Ridge, Lasso, and Elastic Net Regressions. These techniques helped me to find a balance between overly complex and overly simplistic models. Regularization helped prevent the model from performing well on the training data but relativelly unprecise on the testing data. Moreover, Ridge Regression works well with datasets where most variables have useful predictive value, while Lasso Regression is more appropriate for datasets with many variables with little predictive value. Elastic Net Regression combines the strengths of both techniques and is particularly useful for complex datasets with many variables of unknown predictive value (far too many variables to understand).Finally, I combined the individual predictors using stacking and used Ridge Regression as my final estimator. This allowed me to achieve better performance than using any individual algorithm by itself.