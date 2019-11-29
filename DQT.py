#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import statsmodels.api as sm
import seaborn as sns

from PIL import Image
from tqdm import tqdm_notebook
from copy import deepcopy
from datetime import date
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ### Load Precipitation data

# In[2]:

class NQT():
	def __init__(self):
		pass
	def process(self,filename,N,T):
		discharge_df = pd.read_excel(filename) # Load the discharge data for given location
		discharge_df = discharge_df.reset_index(drop=True)
		discharge_df = discharge_df.fillna(0) # Replace the nan values with 0's
		discharge_df["Year"] = discharge_df.apply(lambda row: self.get_year(row), axis=1)
		discharge_df.head()	
		num_days = N
		num_years = T
		years_list = sorted(discharge_df['Year'].unique().tolist())[-num_years:]
		data_list = []
		for year in years_list:
		    year_df = discharge_df[discharge_df["Year"] == year]
		    discharge_mat = np.array(year_df['Discharge'].tolist())
		    moving_average = np.convolve(discharge_mat, np.ones((num_days,)) / num_days, mode='valid')
		    min_flow = np.min(moving_average)
		    data_list.append([min_flow, year])
		data_list = sorted(data_list, key=lambda x: x[0])
		for i, row in enumerate(data_list):
		    P = (i + 1) / (num_years + 1)
		    T = 1 / P
		    data_list[i].append(P)
		    data_list[i].append(T)
		data_list = np.array(data_list)
		fig = plt.figure(figsize=(10, 6))
		plt.scatter(data_list[:, 2], data_list[:, 0], c='k')
		plt.plot(data_list[:, 2], data_list[:, 0])
		min_val = max(i[2] for i in data_list if i[2] < (num_years+0.0)/100)
		x_cor = list(data_list[:,2])
		min_index = x_cor.index(min_val)
		x1 = x_cor[min_index]+0.0
		y1 = data_list[:,0][min_index]+0.0
		x2 = x_cor[min_index+1]+0.0
		y2 = data_list[:,0][min_index+1]+0.0
		x_val = (num_years+0.0)/100.0
		y_val = y1 + ((y2-y1)*(x_val-x1)/(x2-x1))
		plt.scatter(x_val,y_val,marker = 'o')
		plt.xlabel('Probability', fontsize=16)
		plt.ylabel('Minimum Flow', fontsize=16)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.title('DQT Plot for D: {}, T: {}, DQT value : ({},{})'.format(num_days, num_years,x_val,y_val), fontsize=20)
		plt.show()
	def get_year(self,row):
    		date = row['Date']
    		return date.year


class NQT_part2():
	def __init__(self):
		pass
	def process(self,filename1,filename2,filename3,filename4,alpha,basin_num,num_days,num_years):
		basin_mat = scipy.io.loadmat('basin.mat')['rev_new']
		limit_mat = scipy.io.loadmat('latlong_limit.mat')['limit']
		basin_daily_precipitation_mat = scipy.io.loadmat('basin_daily_precipitation.mat')['basin_daily_precipitation']
		basin_mat_delineated = scipy.io.loadmat('basin_mat_delineated.mat')['basin_mat_delineated']
		#basin_num = 6
		discharge_df = pd.read_excel(filename1) # Load the discharge data for given location
		discharge_df = discharge_df.reset_index(drop=True)
		discharge_df = discharge_df.fillna(0) # Replace the nan values with 0's
		discharge_df["Year"] = discharge_df.apply(lambda row: self.get_year(row), axis=1)
		discharge_df.head()
		years_list = sorted(discharge_df['Year'].unique().tolist())[-num_years:]
		X, y = self.get_data(discharge_df, basin_num, basin_daily_precipitation_mat, basin_mat_delineated,alpha)
		rf = RandomForestRegressor(n_estimators=100, random_state=42)
		rf.fit(X, y)
		train_score = rf.score(X, y)
		print("R2: {}".format(train_score))
		# ### Update the precipitation data
		new_discharge_df = deepcopy(discharge_df)
		new_discharge_df = new_discharge_df[(new_discharge_df["Year"] >= years_list[0]) & (new_discharge_df["Year"] <= years_list[-1])]
		X, y = self.get_data(new_discharge_df, basin_num, basin_daily_precipitation_mat, basin_mat_delineated,alpha,decrease=True)
		y_pred = rf.predict(X)
		new_discharge_df["New_Discharge"] = y_pred
		new_discharge_df.head()
		data_list = []
		for year in years_list:
		    year_df = new_discharge_df[new_discharge_df["Year"] == year]
		    new_discharge_mat = np.array(year_df['New_Discharge'].tolist())
		    moving_average = np.convolve(new_discharge_mat, np.ones((num_days,)) / num_days, mode='valid')
		    min_flow = np.min(moving_average)
		    data_list.append([min_flow, year])
		data_list = sorted(data_list, key=lambda x: x[0])
		for i, row in enumerate(data_list):
			P = (i + 1) / (num_years + 1)
			T = 1 / P
			data_list[i].append(P)
			data_list[i].append(T)
		data_list = np.array(data_list)
		fig = plt.figure(figsize=(10, 6))
		plt.scatter(data_list[:, 2], data_list[:, 0], c='k')
		plt.plot(data_list[:, 2], data_list[:, 0])
		min_val = max(i[2] for i in data_list if i[2] < (num_years+0.0)/100)
		x_cor = list(data_list[:,2])
		min_index = x_cor.index(min_val)
		x1 = x_cor[min_index]+0.0
		y1 = data_list[:,0][min_index]+0.0
		x2 = x_cor[min_index+1]+0.0
		y2 = data_list[:,0][min_index+1]+0.0
		x_val = (num_years+0.0)/100.0
		y_val = y1 + ((y2-y1)*(x_val-x1)/(x2-x1))
		plt.scatter(x_val,y_val,marker = 'o')
		plt.xlabel('Probability', fontsize=16)
		plt.ylabel('Minimum Flow', fontsize=16)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.title('DQT Plot for D: {}, T: {}, alpha : {}, DQT value : ({},{})'.format(num_days, num_years, alpha, x_val,y_val), fontsize=20)
		plt.show()

	def get_data(self,discharge_df, basin_num, basin_daily_precipitation_mat, basin_mat_delineated,alpha,decrease=False):
	    discharge_mat = np.array(discharge_df['Discharge'].tolist()) # Get the runoff values
	    precipitation_start_date = date(1901, 1, 1) # Start date of precipitation data given to us
	    start_date = discharge_df.iloc[0]['Date'] # The start date of discharge data for given location
	    year, month, day = start_date.year, start_date.month, start_date.day
	    discharge_start_date = date(year, month, day) # Convert to datetime format
	    
	    end_date = discharge_df.iloc[-1]['Date'] # The start date of discharge data for given location
	    year, month, day = end_date.year, end_date.month, end_date.day
	    discharge_end_date = date(year, month, day) # Convert to datetime format
	    
	    x_start = (discharge_start_date - precipitation_start_date).days # Starting index for precipitation data
	    x_end = (discharge_end_date - precipitation_start_date).days + 1 # Ending index for precipitation data
	    
	    # Flatten all the given data to perform masking operation
	    basin_daily_precipitation_mat = basin_daily_precipitation_mat.reshape(-1, basin_daily_precipitation_mat.shape[2])
	    basin_mat_delineated = basin_mat_delineated.reshape(-1)
	    
	    # Create a mask array and select only those cells that belong to the given watershed
	    # Watershed here is defined by the basin_num 
	    mask_array = (basin_mat_delineated == basin_num)
	    masked_data = basin_daily_precipitation_mat[mask_array, x_start:x_end].T
	    if decrease:
	        masked_data = masked_data * (1-(alpha/100))
	    return masked_data, discharge_mat

	def get_year(self,row):
		date = row['Date']
		return date.year