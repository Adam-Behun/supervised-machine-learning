import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
train = pd.read_csv('../data/train.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# Create and train the models
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_reg.fit(X_train, y_train)

xgb_reg = XGBRegressor(n_estimators=100, random_state=42)
xgb_reg.fit(X_train, y_train)

# Evaluate the models on the testing set
lin_reg_preds = lin_reg.predict(X_test)
lin_reg_rmse = np.sqrt(mean_squared_error(y_test, lin_reg_preds))
print('Linear Regression RMSE:', lin_reg_rmse)

rf_reg_preds = rf_reg.predict(X_test)
rf_reg_rmse = np.sqrt(mean_squared_error(y_test, rf_reg_preds))
print('Random Forest RMSE:', rf_reg_rmse)

gb_reg_preds = gb_reg.predict(X_test)
gb_reg_rmse = np.sqrt(mean_squared_error(y_test, gb_reg_preds))
print('Gradient Boosting RMSE:', gb_reg_rmse)

xgb_reg_preds = xgb_reg.predict(X_test)
xgb_reg_rmse = np.sqrt(mean_squared_error(y_test, xgb_reg_preds))
print('XGBoost RMSE:', xgb_reg_rmse)

# Save the predictions to CSV files
lin_reg_preds_df = pd.DataFrame(lin_reg_preds, columns=['predictions'])
lin_reg_preds_df.to_csv('lin_reg_preds.csv', index=False)

rf_reg_preds_df = pd.DataFrame(rf_reg_preds, columns=['predictions'])
rf_reg_preds_df.to_csv('rf_reg_preds.csv', index=False)

gb_reg_preds_df = pd.DataFrame(gb_reg_preds, columns=['predictions'])
gb_reg_preds_df.to_csv('gb_reg_preds.csv', index=False)

xgb_reg_preds_df = pd.DataFrame(xgb_reg_preds, columns=['predictions'])
xgb_reg_preds_df.to_csv('xgb_reg_preds.csv', index=False)
