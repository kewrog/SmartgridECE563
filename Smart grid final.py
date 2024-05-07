#!/usr/bin/env python
# coding: utf-8

# # PROJECT - ARTIFICIAL INTELLIGENCE IN SMART GRID

# ## NAME -Roger Kewin Samson
# ## HAWK ID - A20563057
# ## MAIL ID - rkewin@hawk.iit.edu
# ## SPRING 2024

# # ILLINOIS INSTITUTE OF TECHNOLOGY 

# # BRIEF OVERVIEW: 

# The final project is to predict the electrical load values using temperature data across various zones. these predictions are crucial for efficient energy management and will be accurate because they are based on the zones' temperature.
# We have the dataset that includes temperature readings for 11 stations and load values for the 20 zones all these were in hour differences. 
# 
# Data preparation : 
# In the given temperature and load values we need to remove the null values and outliers in it. I visualized null data in the load to verify and remove it perfectly. By doing this we can get the cleaned data of both temperature and load we can save it in a data frame or export it as a CSV file. This phase also involves mapping each zone to the most relevant temperature station based on historical data to ensure the accuracy of temperature data used for each zone.
# Mapping the station to zones by correlation makes the matter to get the prediction better. After mapping we merged the data (cleaned_Load_history_final.csv and cleaned_Tem_hiatory_final.csv )according to the mapped station as merged data. We are going to use this merged data for the training and make predictions.
# e began by meticulously organizing and visualizing the essential data, creating a comprehensive table that includes both station ID and zone ID. Following this, we meticulously divided the dataset into three distinct subsets: training, validation, and testing. During this process, we ensured the integrity of the validation set and verified its dimensions.
# 
# Moving forward, we embarked on selecting a suitable machine learning algorithm to ascertain the accuracy of predictions. Leveraging the merged dataset, we meticulously evaluated the chosen algorithm's efficacy in accurately forecasting based on key features such as temperature and station mapping.
# 
# To deepen our understanding, we meticulously examined the top 10 prediction errors stemming from this algorithm. This meticulous analysis provides invaluable insights into the algorithm's strengths and weaknesses.
# 
# In our quest for optimization, we then ventured into employing a more intricate and sophisticated algorithm. Following the same rigorous methodology, we assessed its performance against the dataset, ensuring a thorough comparison with the initial algorithm's results. This meticulous approach allows us to select the most appropriate algorithm for our predictive modeling endeavors.
# After all this the predicted file for the first week of June 2008 will be exported as CSV file.
# 
# 
# 
# 
# 
#  
# 

# ## Making the data cleaned by removing null values , outliers and exploring the data: 

# In this  we removed all null data of overall dataset and outliers and stored it as cleaned data to use it later below.

# In[113]:


import pandas as pd
from scipy import stats
import numpy as np

# Load the datasets
load_data = pd.read_csv('../Load_history_final.csv')
temp_data = pd.read_csv('../Temp_history_final.csv')

# Remove rows where any cell has a missing value
load_data_clean = load_data.dropna()
temp_data_clean = temp_data.dropna()

# Define columns that contain hourly data to check for outliers and zeros
load_hourly_columns = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12',
                       'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24']
temp_hourly_columns = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12',
                       'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24']

# Remove rows with zero values in any of the hourly columns
load_data_clean = load_data_clean[~(load_data_clean[load_hourly_columns] == 0).any(axis=1)]
temp_data_clean = temp_data_clean[~(temp_data_clean[temp_hourly_columns] == 0).any(axis=1)]

# Remove outliers in load data
z_scores_load = np.abs(stats.zscore(load_data_clean[load_hourly_columns]))
load_data_clean = load_data_clean[(z_scores_load < 3).all(axis=1)]

# Remove outliers in temperature data
z_scores_temp = np.abs(stats.zscore(temp_data_clean[temp_hourly_columns]))
temp_data_clean = temp_data_clean[(z_scores_temp < 3).all(axis=1)]

# Save the cleaned data
load_data_clean.to_csv('cleaned_Load_history_final.csv', index=False)
temp_data_clean.to_csv('cleaned_Temp_history_final.csv', index=False)


# The cleaned dataset has been described to analyze and printed.

# In[114]:


import pandas as pd
load_data = pd.read_csv('cleaned_Load_history_final.csv')
temp_data = pd.read_csv('cleaned_Temp_history_final.csv')

load_data_descr= load_data.describe()
temp_data_descr = load_data.describe()

# Print the descriptive statistics
print("Load Data Description:")
print(load_data_descr)
print("\nTemperature Data Description:")
print(temp_data_descr)

# Display the first few rows of each dataset
print("\nLoad Data - First Few Rows:")
print(load_data.head())
print("\nTemperature Data - First Few Rows:")
print(temp_data.head())


# 
# 
# ## The cleaned data has merged for the correlation to map station: 

# In[115]:


import pandas as pd

load_data = pd.read_csv('cleaned_Load_history_final.csv')
temp_data = pd.read_csv('cleaned_Temp_history_final.csv')

# Reshaping the data to long format
temp_long = temp_data.melt(id_vars=['station_id', 'year', 'month', 'day'], var_name='hour', value_name='temperature')
load_long = load_data.melt(id_vars=['zone_id', 'year', 'month', 'day'], var_name='hour', value_name='load')

# Converting 'hour' column from string to integer for merging
temp_long['hour'] = temp_long['hour'].str.extract('(\d+)').astype(int)
load_long['hour'] = load_long['hour'].str.extract('(\d+)').astype(int)

# Merging the data on year, month, day, and hour
merged_data = pd.merge(temp_long, load_long, on=['year', 'month', 'day', 'hour'])

# Display the first few rows of the merged data to verify
print(merged_data.head())


# ## The correlation for the merged data: 

# Grouping and Correlation Calculation:
# 
# 
# First, it groups the data by station.
# Then, for each station, it calculates a number called "correlation" between temperature and electricity usage. This number tells us how much they're related. If it's high, it means when it's hotter, more electricity is used, and vice versa.
# 
# 
# The code organizes these correlation numbers into a list.
# It sorts this list from stations with the highest correlation to the lowest. So, at the top are stations where temperature and electricity usage are strongly connected.
# 
# 
# 
# Finally, it shows us this list, with each station's ID and its correlation number. This helps us see which stations are most influenced by temperature changes in terms of electricity usage.
# 
# 

# In[116]:


# Group by station_id and calculate correlation
correlation_data = merged_data.groupby('station_id').apply(lambda x: x['temperature'].corr(x['load']))

# Convert to DataFrame and sort by correlation
correlation_df = correlation_data.reset_index()
correlation_df.columns = ['station_id', 'correlation']
correlation_df = correlation_df.sort_values(by='correlation', ascending=False)

# Display the correlations
print(correlation_df)


# The code starts by grouping the data by both the zone and the station. This way, it organizes the information based on where the stations are located and then breaks it down further by each individual station within each zone. Then, it calculates something called "correlation" for each combination of zone and station. This number tells us how closely connected the temperature and electricity usage are at each station within each zone. If the correlation is high, it means that when it's hotter, more electricity is used, and when it's cooler, less electricity is used.
# After calculating the correlation for each zone-station combination, the code puts these correlation numbers into a DataFrame called correlation_df. This DataFrame has columns for the zone ID, the station ID, and the correlation value. Then, it sorts this DataFrame to find the stations with the highest correlation for each zone. This helps us identify which stations in each zone have the strongest connection between temperature and electricity usage.
# 
# 
# The code prints out the DataFrame best_stations. This DataFrame contains the best station (the one with the highest correlation) for each zone. By looking at this information, we can see which stations within each zone are most influenced by changes in temperature when it comes to electricity usage.
# 
# 
# 
# 
# 
# 
# 

# In[117]:


import pandas as pd


# Calculate the correlation for each station and zone combination
grouped = merged_data.groupby(['zone_id', 'station_id'])
correlation_by_zone_station = grouped.apply(lambda x: x['temperature'].corr(x['load']))

# Convert the series to a DataFrame
correlation_df = correlation_by_zone_station.reset_index()
correlation_df.columns = ['zone_id', 'station_id', 'correlation']

# Find the station with the highest correlation for each zone
best_stations = correlation_df.loc[correlation_df.groupby('zone_id')['correlation'].idxmax()]

# Display the best station for each zone
print(best_stations)


# # Table showing mapping of a temperature station for each load zone: 

# | zone_id  |  station_id  | 
# |----------|--------------|
# |
# |1  | 8 |
# |2 | 8 |
# |3 | 1 |
# |4 | 8 |
# |5 | 8 |
# |6 | 1 |
# |7 | 1 |
# |8 | 8 |
# |9 | 8 |
# |10 | 8 |
# |11 | 1 |
# |12 | 8 |
# |13 | 8 |
# |14 | 1 |
# |15 | 3 |
# |16 | 8 |
# |17 | 1 |
# |18 | 1 |
# |19 | 2 |
# |20 | 8 |
#   

# ## mapping station to zones and exporting it as a CSV file: 

# A dictionary named zone_station_mapping is created, which pairs each zone with its corresponding station ID. For instance, zone 1 is linked to station 8, zone 2 to station 8, and so forth. This mapping helps in associating each zone with the appropriate station where data will be collected or analyzed.
# 
# The load_long DataFrame's 'zone_id' column is linked to the respective 'station_id' using the map() function along with the zone_station_mapping dictionary. This process assigns each record in the load_long DataFrame to the correct station ID based on its zone.
# Subsequently, the load_long DataFrame is merged with the temp_long DataFrame utilizing common columns such as 'station_id', 'year', 'month', 'day', and 'hour'. This merging operation combines data from both DataFrames into a single DataFrame referred to as merged_data.
# Then the dataframe data is exported as a csv file.

# In[118]:


# Mapping zones to stations
zone_station_mapping = {
    1: 8, 2: 8, 3: 1, 4: 8, 5: 8, 6: 1, 7: 1, 8: 8, 9: 8, 10: 8,
    11: 1, 12: 8, 13: 8, 14: 1, 15: 3, 16: 8, 17: 1, 18: 1, 19: 2, 20: 8
}
load_long['station_id'] = load_long['zone_id'].map(zone_station_mapping)
merged_data = pd.merge(load_long, temp_long, on=['station_id', 'year', 'month', 'day', 'hour'])


# In[119]:


import pandas as pd


merged_data.to_csv('merged_data.csv', index=False)  # This will save the DataFrame to a CSV file named 'filename.csv'


# # Python code that splits the load and temperature datasets into two subsets: training/validation, and test. . 

# We are loading the merged_data file  and preparing the features and target variable for the training and make it ready. Now as per the delieverables we split our data in to the 70% for training and 30% for testing  and printed the shape of the data.
# 
# RANDOM_STATE:
# 
# I chose "0" as a random_state because i dont want make differ while iteration to make the same each time i chose  the 0 as the random state for the split

# In[120]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv('merged_data.csv')

# Prepare features and target variable
X = data[['zone_id', 'year', 'month', 'day', 'hour', 'station_id', 'temperature']]
y = data['load']

# First split to create train/validation and test sets for load and temperature datasets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Print the shapes
print("Training/Validation set shapes:")
print("X_train_val shape:", X_train_val.shape)
print("y_train_val shape:", y_train_val.shape)
print("\nTest set shapes:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# # Python code that uses an additional split to create a validation dataset 

# Then we created the another split for the validation as the deliverables need as the additional split. we imported the merged_data.csv file as we done above then trained data is splitted in to train and validation.printed the shape of the dataset.

# Random_State:
# 
# I chose as the same above "0" as the random_state value because i dont want different values to be done in the iteration hence i chose "0" as the random_state value.

# In[121]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv('merged_data.csv')

# Prepare features and target variable
X = data[['zone_id', 'year', 'month', 'day', 'hour', 'station_id', 'temperature']]
y = data['load']

# First split to create train/validation and test sets for load and temperature datasets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Additional split to create validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=0) # 0.25 x 0.8 = 0.2

# Print the shapes
print("Training set shapes:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("\nValidation set shapes:")
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# # Procedure documenting your design process and the tradeoffs you considered in building your first machine learning regressor model.

# Data Handling and Setup:
# 
# I loaded the dataset from 'merged_data.csv' using Pandas and organized it into features (labeled as 'X') and the target variable (labeled as 'y').
# 
# Data Scaling:
# 
# To ensure consistency in scale across all features, I standardized the target variable using StandardScaler.
# 
# Data Partitioning:
# 
# Using the train_test_split function, I divided the data into training, validation, and test sets. Initially, I allocated 30% of the data for testing and validation, and then split the remaining data equally between validation and test sets.
# 
# Feature Standardization:
# 
# All features were standardized by scaling them to have a mean of zero and a variance of one using StandardScaler.
# 
# Model Setup and Training:
# 
# I initialized a RandomForestRegressor with specific parameters: 50 trees (n_estimators), a minimum of 10 samples per leaf (min_samples_leaf), and a fixed random state (random_state=0) for reproducibility. The model was trained using the scaled training data.
# 
# Model Assessment:
# 
# Predictions were made on the training, validation, and test sets. Performance was evaluated using Mean Squared Error (MSE) to gauge the average squared difference between predicted and actual values, and R-squared (R²) to measure the proportion of variance explained by the model.
# 
# Performance Examination:
# 
# I checked the model performance across training, validation, and test sets, examining any tradeoffs between them. Additionally, I considered discrepancies in performance across different datasets.compared to others this was the best algorithm gives best accuracy.

# In[122]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load your data
data = pd.read_csv('merged_data.csv')

# Prepare features and target variable
X = data[['zone_id', 'year', 'month', 'day', 'hour', 'station_id', 'temperature']]
y = data['load']

# Normalize the target variable
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Split the data into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_scaled, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Feature scaling for features
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_val_scaled = scaler_x.transform(X_val)
X_test_scaled = scaler_x.transform(X_test)

# Initialize and train the RandomForestRegressor with reduced complexity
rf_model = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=0)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = rf_model.predict(X_train_scaled)
y_val_pred = rf_model.predict(X_val_scaled)
y_test_pred = rf_model.predict(X_test_scaled)

# Calculate performance metrics
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print the performance metrics
print("Training MSE:", train_mse)
print("Validation MSE:", val_mse)
print("Test MSE:", test_mse)
print("Training R²:", train_r2)
print("Validation R²:", val_r2)
print("Test R²:", test_r2)


# ## OBSERVATIONS: 

# First i chose Linear regression and it is low accuracy then i choose the Randomforest regressor because it makes high accuracy score compared to that.
# I scaled the target variable using standardScaler.This ensures that all feature have the same scale which improve the accuracy score of the model.
# 
# Feature Scaling: You standardized the features using StandardScaler. By centering the features around zero and scaling them to have a unit variance, you make the optimization process smoother and prevent certain features from dominating others, leading to potentially better model performance.
# 
# I specified hyperparameters as n_estimators and the min_samples_leaf. I keep on changing value by uing 100 estimators without min_sample_leaf it gave very high mse score . by tuning and make it 50 and 10 makes the accuracy perfect scores.
# by doing all this evaluation gives best score in mse and r^2 score without any overfitting or underfiiting.
# 

# # PREDICTION AND EXPORTING CSV FILE FOR MY BEST ALGORITHM: 

# To make accurate predictions about the load for June2008, we started by matching temperature statios to their respective load zones. we achieved this by using a mapping dictionary to assign each temperature stations to its corresponding load zone. Any missing or invalid mappings were removed from consideration to ensure the accuracy of our predictions.
# 
# Once we had mapped the temperature stations to their load zones, we combined this mapping with our temperature data and focused on observations from the first week of June 2008. This allowed us to narrow down our dataset to the specific timeframe we were interested in.
# 
# With our data properly filtered and organized, we prepared the temperature feature for prediction. We isolated the temperature readings, which would serve as our input for predicting load values.
# 
# Using a Random Forest model that we had previously trained, we made predictions about the load values for June 2008 based on the temperature data. This model had learned patterns from historical data and was now capable of making predictions about future load values.
# 
# The predicted load values were then incorporated back into our dataset. This step allowed us to see both the actual temperature readings and the corresponding predicted load values side by side.
# 
# Finally, we formatted these predictions for easy analysis and presentation. We structured the data to resemble the format of a CSV file called 'Load_Prediction.csv', making it convenient for further examination or sharing with others.
# 
# By following this process, we ensured that our predictions for the load in June 2008 were well-prepared and ready for use in decision-making or further analysis.

# In[123]:


june_2008_temp.loc[:, 'zone_id'] = june_2008_temp['station_id'].map(zone_station_mapping_reversed)
june_2008_temp = june_2008_temp.dropna(subset=['zone_id'])  # Drop rows where zone_id is NaN after mapping
import pandas as pd

# Create a DataFrame from your mapping
zone_station_df = pd.DataFrame({
    'zone_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'station_id': [8, 8, 1, 8, 8, 1, 1, 8, 8, 8, 1, 8, 8, 1, 3, 8, 1, 1, 2, 8]
})

# Merge this mapping with the temperature data to assign the correct station's temperature to each zone
june_2008_temp = pd.merge(zone_station_df, temp_long, on='station_id')

# Filter for the first week of June 2008
june_2008_temp = june_2008_temp[(june_2008_temp['year'] == 2008) & (june_2008_temp['month'] == 6) & (june_2008_temp['day'] <= 7)]

# Prepare features for prediction
X_june_2008 = june_2008_temp[['temperature']]
# Predict the load for June 2008 using the trained Random Forest model
predicted_loads_rf = random_forest_model.predict(X_june_2008)

# Add predictions back to the DataFrame
june_2008_temp['predicted_load'] = predicted_loads_rf

predicted_output_rf = june_2008_temp.pivot_table(index=['zone_id', 'year', 'month', 'day'], columns='hour', values='predicted_load', aggfunc='first')
predicted_output_rf.columns = [f'h{col}' for col in predicted_output_rf.columns]
predicted_output_rf.reset_index(inplace=True)

predicted_output_rf.to_csv('Load_Prediction.csv', index=False)


# In[124]:


import pandas as pd

# Assuming 'test_data' includes the 'zone', 'year', 'month', 'day', 'hour', and 'load' (true load),
# and y_test_pred_rf contains the predicted loads from the Random Forest model.

# Add the predicted loads to the test data DataFrame
test_data['predicted_load'] = y_test_pred_rf

# Calculate the relative percentage error
test_data['relative_percentage_error'] = 100 * (test_data['load'] - test_data['predicted_load']) / test_data['load']

# Handle cases where true load is zero to avoid infinite values
test_data['relative_percentage_error'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)

# Drop rows where relative percentage error could not be calculated (e.g., true load is zero)
test_data.dropna(subset=['relative_percentage_error'], inplace=True)

# Sort the DataFrame by the absolute value of the relative percentage error in descending order
test_data['abs_relative_error'] = test_data['relative_percentage_error'].abs()
sorted_errors = test_data.sort_values(by='abs_relative_error', ascending=False)

top_10_errors = sorted_errors.head(10)[['zone_id', 'year', 'month', 'day', 'hour', 'predicted_load', 'load', 'relative_percentage_error']]

# Output the top 10 errors
print(top_10_errors)


#  we loaded the dataset ('merged_data.csv') containing information about load values and associated features. Our objective was to build a regressor model to predict load categories.
# 
# Data Preprocessing:
# 
# As a preprocessing step, we converted the 'load' feature into binary categories based on its median value. This transformation allowed us to simplify the classification task, making it more suitable for logistic regression.
# 
# Feature Selection and Target Variable Preparation:
# 
# We selected relevant features such as 'zone_id', 'year', 'month', 'day', 'hour', 'station_id', and 'temperature' to be used as predictors ('X') for our model. The target variable ('y') was defined as the binary load category.
# 
# Data Splitting:
# 
# The dataset was split into training, validation, and test sets using the train_test_split function from sklearn.model_selection. This splitting strategy helped us assess the model's performance on unseen data while ensuring that the training process was robust.
# 
# Feature Scaling:
# 
# Since logistic regression is sensitive to the scale of features, we standardized the feature values using StandardScaler from sklearn.preprocessing. This scaling technique ensured that all features had a similar influence on the model, thereby preventing any particular feature from dominating the others.
# 
# Model Initialization and Training:
# 
# We initialized a logistic regression model (LogisticRegression) with a maximum iteration limit of 1000 to avoid convergence issues. This choice of algorithm was made considering its simplicity.
# 
# Model Evaluation:
# 
# After training the model, we made predictions on the training, validation, and test sets to evaluate its performance. The accuracy score, calculated using accuracy_score from sklearn.metrics, was chosen as the evaluation metric to measure the proportion of correctly predicted instances.
# 
# While logistic regression offers simplicity and interpretability, it may not capture complex nonlinear relationships present in the data. Additionally, it assumes linearity between features and the log-odds of the target variable, which might not always hold true.

# I chose random_state value = '0' because of its constant, while iteration i dont want to get different value. So, I chose 0.

# In[125]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load your data
data = pd.read_csv('merged_data.csv')

# Convert 'load' to binary categories based on the median value
median_load = data['load'].median()
data['load_category'] = (data['load'] > median_load).astype(int)

# Prepare features and target variable
X = data[['zone_id', 'year', 'month', 'day', 'hour', 'station_id', 'temperature']]
y = data['load_category']

# Split the data into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize and train logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_train_pred = log_reg.predict(X_train_scaled)
y_val_pred = log_reg.predict(X_val_scaled)
y_test_pred = log_reg.predict(X_test_scaled)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the accuracy scores
print("Training Accuracy:", train_accuracy)
print("Validation Accuracy:", val_accuracy)
print("Test Accuracy:", test_accuracy)


# We used the trained logistic regression model (log_reg) to predict load categories for the test data (X_test_scaled). These predictions were stored in y_test_pred.
# 
# 
# Calculating Predicted Loads:
# Since logistic regression predicts probabilities, we extracted the probability of the positive class (i.e., the probability of load being high) using predict_proba. This was done by selecting the second column of the predicted probabilities, denoted as [:, 1], and stored in predicted_loads_test.
# 
# We calculated the relative percentage error between the true load values (y_test) and the predicted load probabilities (predicted_loads_test). This error was expressed as a percentage and stored in relative_percentage_error.
# 
# To organize and analyze the results, we created a DataFrame (test_results) containing the original test data (X_test) along with the predicted loads (predicted_loads_test), true load values (y_test), and relative percentage errors (relative_percentage_error).We sorted the test results DataFrame based on the magnitude of the relative percentage error to identify the top 10 errors. This was achieved by reindexing the DataFrame with indices sorted in descending order of the absolute error values.
# Finally, we printed the top 10 errors, including relevant information such as zone ID, date and time, predicted load, true load, and relative percentage error.

# In[126]:


import numpy as np

# Make predictions on the test data
y_test_pred = log_reg.predict(X_test_scaled)

# Calculate predicted loads
predicted_loads_test = log_reg.predict_proba(X_test_scaled)[:, 1]

# Calculate relative percentage error
relative_percentage_error = 100 * (y_test - predicted_loads_test) / y_test

# Create DataFrame for test data with predictions and errors
test_results = X_test.copy()
test_results['predicted_load'] = predicted_loads_test
test_results['true_load'] = y_test
test_results['relative_percentage_error'] = relative_percentage_error

# Sort by the magnitude of relative percentage error
top_10_errors = test_results.reindex(np.abs(test_results['relative_percentage_error']).sort_values(ascending=False).index).head(10)

# Display the top 10 errors
print(top_10_errors[['zone_id', 'year', 'month', 'day', 'hour', 'predicted_load', 'true_load', 'relative_percentage_error']])


