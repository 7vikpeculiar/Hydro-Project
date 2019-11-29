import matplotlib 
matplotlib.use('TkAgg')
from tkinter import ttk
from tkinter import * 
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os,sys, time, random, math, numpy as np, scipy.io, statsmodels.api as sm, seaborn as sns
from tkinter.filedialog import askopenfilename,askdirectory
import pandas as pd
import numpy as np
from glob import glob as glb
from PIL import Image
from tqdm import tqdm_notebook
from copy import deepcopy
from datetime import date
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from DQT import NQT,NQT_part2

from scipy.io import loadmat
from scipy.stats import gamma
from scipy.stats import norm
from matplotlib.figure import Figure
import numpy as np

import shapefile as shp  # Requires the pyshp package
import geopandas
from shapely.geometry import Polygon
from shapely.geometry import mapping
import geopandas as gpd

# from code import get_SPI
if sys.version_info[0] < 3:
	import Tkinter as Tk
else:
	import tkinter as Tk

filename5 = ''
filename6 = ''
filename7 = ''
filename8 = ''
root = Tk.Tk()
root.geometry('900x900')
root.wm_title("Team 2 - HSMD Tool")
root.configure(background="light green")

root.style = Style()
# Style.configure("TFrame",background="light green")
s = Style()
s.configure('new.TFrame', background='#7AC5CD')

TAB_CONTROL = Notebook(root)
############
TAB1 = Tk.Frame(TAB_CONTROL,width=900,height=900,background="light green")
# TAB1.config(background="light green")
# TAB1.config(background='red')
# TAB1.style.theme_use("clam")
 
TAB_CONTROL.add(TAB1, text='Task 1')
#Tab2
TAB2 = Tk.Frame(TAB_CONTROL,width=900,height=900,background="light green")
# TAB1.style.theme_use("clam")
 
TAB_CONTROL.add(TAB2, text='Task 2')
 
TAB3 = Tk.Frame(TAB_CONTROL,width=900,height=900,background="light green")
# TAB3.style.theme_use("clam")
 
TAB_CONTROL.add(TAB3, text='Task 3')
#Tab2
TAB4 = Tk.Frame(TAB_CONTROL,width=900,height=900,background="light green")
# TAB4.style.theme_use("clam")
 
TAB_CONTROL.add(TAB4, text='Bonus')
 
TAB_CONTROL.grid(row = 0)
############
# TAB1 = Frame(TAB_CONTROL,width=900,height=900,style='new.TFrame')
# # TAB1.config(background="light green")
# # TAB1.config(background='red')
# # TAB1.style.theme_use("clam")

# TAB_CONTROL.add(TAB1, text='Tab 1')
# #Tab2
# TAB2 = Frame(TAB_CONTROL,width=900,height=900)
# # TAB1.style.theme_use("clam")

# TAB_CONTROL.add(TAB2, text='Tab 2')

# TAB3 = Frame(TAB_CONTROL,width=900,height=900)
# # TAB3.style.theme_use("clam")

# TAB_CONTROL.add(TAB3, text='Tab 3')
# #Tab2
# TAB4 = Frame(TAB_CONTROL,width=900,height=900)
# # TAB4.style.theme_use("clam")

# TAB_CONTROL.add(TAB4, text='Tab 4')

TAB_CONTROL.grid(row = 0)

root.style.theme_use("clam")

style = Style() 
style.configure('TButton', font = 
			   ('calibri', 10, 'bold'), 
					borderwidth = '2',bg='green') 

mygreen = "#d2ffd2"
myred = "#dd0202"

style.theme_create( "yummy", parent="alt", settings={
		"TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
		"TNotebook.Tab": {
			"configure": {"padding": [5, 1], "background": mygreen },
			"map":       {"background": [("selected", myred)],
						  "expand": [("selected", [1, 1, 1, 0])] } } } )
style.theme_use("yummy")

# Changes will be reflected 
# by the movement of mouse. 
# style.map('TButton', foreground = [('active', '! disabled', 'green')], 
					 # background = [('active', 'black')]) 
  

def process_file(N,T):
	N = int(N)
	T = int(T)
	file = askopenfile(mode ='r', filetypes =[('Excel Sheets', '*.xls')])
	if file is not None and N!=0 and T!=0:
		obj = NQT()
		obj.process(file.name,N,T)
	return


def _quit():
	root.quit()     
	root.destroy()

L0 = Label(TAB3, text="Part1-Find DQT", font='TkHeadingFont 20 bold')
# L0.pack(side=TOP,pady=5)
L0.grid(row = 0, column = 4, padx = 3, pady = 10)

#part1
L1 = Label(TAB3, text="Value of N", font="Times 15 bold")
# L1.pack(side=TOP,pady=5)
L1.grid(row = 1, column = 1, pady = 4)
E1 = Entry(TAB3)
E1.grid(row = 1, column = 4, pady = 4)
# E1.pack(pady=5)

L2 = Label(TAB3, text="Value of T", font="Times 15 bold")
L2.grid(row = 2, column = 1, pady = 4)
# L2.pack(side=TOP,pady=5)
E2 = Entry(TAB3)
E2.grid(row = 2, column = 4, pady = 4)
# E2.pack(pady=10)

# btn = Button(TAB3, text ='Upload xls Data', style = 'NuclearReactor.TButton', command = lambda: process_file(E1.get(),E2.get())) 
# btn.pack(side = TOP, pady = 5)
# btn = Tk.Button(TAB3, text ='Upload xls Data', font = "Times 10 bold",background = 'red',  borderwidth='3',command = lambda: process_file(E1.get(),E2.get())) 
# btn.grid(row = 10, column = 4, pady = 8)
#part2
def process_file1():
	global filename1
	file = askopenfile(mode ='r', filetypes =[('Excel Sheets', '*.xls')])
	if file is not None:
		filename1=file.name
def process_file2():
	global filename2
	file = askopenfile(mode ='r', filetypes =[('mat', '*.mat')])
	if file is not None:
		filename2=file.name
def process_file3():
	global filename3
	file = askopenfile(mode ='r', filetypes =[('mat', '*.mat')])
	if file is not None:
		filename3=file.name
def process_file4(N,T,basin_num,alpha):
	N = int(N)
	T = int(T)
	basin_num = int(basin_num)
	alpha = float(alpha)
	global filename4
	file = askopenfile(mode ='r', filetypes =[('mat', '*.mat')])
	if file is not None:
		filename4=file.name
	if N!=0 and T!=0 and filename1 is not None and filename2 is not None and filename3 is not None and filename4 is not None:
		print("PROCESSING.....")
		obj = NQT_part2()
		obj.process(filename1,filename2,filename3,filename4,alpha,basin_num,N,T)

def process_file5():
	global filename5
	file_ = askopenfilename(filetypes =[('Shape Files', '*.shp')])
	if file_ is not None:
		filename5 = file_
	print(filename5)

def process_file6():
	global filename6
	file_ = askopenfilename(filetypes =[('Historic Precipitation .mat files', '*.mat')])
	if file_ is not None:
		filename6 = file_
	print(filename5)

def bonus_part(file_,time=3000):
	x = loadmat(file_)
	matrix = x['basin_daily_precipitation']
	shp = matrix.shape
	final_output = np.zeros([shp[0],shp[1]])
	for i in range(shp[0]):
		for j in range(shp[1]):
			single_point_slice = matrix[i,j,:]
			single_point_slice = [single_point_slice[i] for i in range(0,len(single_point_slice),time)]
			if all(i >= 0 for i in single_point_slice):
				output = []
				for ele in single_point_slice:
					if ele > 0:
						output.append(ele)
					else:
						output.append(1e-3)
				single_point_slice = output
				fit_alpha, fit_loc, fit_beta =gamma.fit(single_point_slice)
				out = gamma.pdf(single_point_slice,a=fit_alpha)

				x = [ele for ele in range(0, len(single_point_slice))]
				s = np.std(out)
				m = np.mean(out)
				out = norm.pdf(out,m,s)
				final_output[i][j] = out[-1]
	
	plt.imshow(final_output, extent=[0, 1, 0, 1])
	plt.colorbar()
	plt.show()    

def vector_to_threshold(file_name,area_threshold = 0.05):
	gdf = gpd.read_file(file_name)
	minx, miny, maxx, maxy = gdf.geometry.total_bounds

	left = minx -0.5 
	right = maxx +0.5
	up = maxy +0.5
	down = miny -0.5
	range_ = 0.25
	area_threshold = 0.05
	g = gdf.geometry[0]
	arr1 = arange(left, right,range_)
	arr2 = arange(down, up, range_)

	mask = np.zeros(shape=(len(arr1),len(arr2)))
	for i in range(len(arr1)-1):
		for j in range(len(arr2)-1):
			pgon1 = Polygon([[arr1[i],arr2[j]],
							[arr1[i],arr2[j+1]], 
							[arr1[i+1],arr2[j+1]], 
							[arr1[i+1], arr2[j]  ]
							])
			intersection = g.intersection(pgon1)
			if(intersection.area/pgon1.area) > area_threshold: 
				mask[i][j] = 1
			else:
				mask[i][j] = 0

	return np.flip(np.transpose(mask),0)
	
def run_part1(file_,threshold):
	threshold = 0.25
	val = []
	if file_ is not None: 
		sf = shp.Reader(file_)
		for shape in sf.shapeRecords():
			x = [i[0] for i in shape.shape.points[:]]
			y = [i[1] for i in shape.shape.points[:]]
		val.append((x,y))
		val.append(vector_to_threshold(file_,threshold))
	f = plt.figure(figsize=(5, 4), dpi=100)
	a = f.add_subplot(121)
	a.plot(val[0][0],val[0][1])
	b = f.add_subplot(122)
	b.matshow(val[1], extent=[0, 1, 0, 1])	
	plt.show()

def run_bonus(file_, time):
	print(file_)
	print(time)
	bonus_part(file_,int(time))

def get_SRI(time):
	final_out = []
	files = glb('*.xls')
	print(files)
	for file in files:
		df = pd.read_excel(file)
		arr = df['Gauge'].to_list()
		arr = [arr[i] for i in range(0,len(arr),time)]
		if True:
			temp = []
			for ele in arr:
				if(ele <= 0):
					temp.append(1e-3)
				else:
					temp.append(ele)
			arr = temp
			fit_alpha, fit_loc, fit_beta =gamma.fit(arr)
			arr = gamma.pdf(arr,a=fit_alpha)
			x = [ele for ele in range(0, len(arr))]
			s = np.std(x)
			m = np.mean(x)
			out = norm.pdf(x,m,s)
		final_out.append(out)
	another_out = []
	min_len = min([len(ele) for ele in final_out])
	print(min_len)
	for ele in final_out:
		another_out.append(ele[:min_len])
	
	plt.imshow(another_out, extent=[0, 1, 0, 1])
	plt.axes()
	plt.colorbar()
	plt.show()

def process_file7():
	global filename7
	file_ = askopenfilename(filetypes =[('DEM Files', '*.tif *.tiff *.mat')])
	if file_ is not None:
		filename7 = file_

def process_file8():
	global filename8
	file_ = askopenfilename(filetypes =[('CSV Files', '*.csv')])
	if file_ is not None:
		filename8 = file_
	if filename7 is not None and filename8 is not None:
		import matplotlib.image as mpimg
		img=mpimg.imread('watershed.png')
		imgplot = plt.imshow(img)
		import time
		time.sleep(4)
		plt.show()


L0 = Label(TAB3, text="Part2-Find PPT change vs DQT", font='TkHeadingFont 20 bold')
L0.grid(row = 5, column = 4, pady = 8)
# L0.pack(side=TOP,pady=5)

L3 = Label(TAB3, text="Value of N", font="Times 15 bold")
# L3.pack(side=TOP,pady=5)
L3.grid(row = 6, column = 1, pady = 8)

E3 = Entry(TAB3)
E3.grid(row = 6, column = 4, pady = 8)

# E3.pack(pady=10)

L4 = Label(TAB3, text="Value of T", font="Times 15 bold")
L4.grid(row = 7, column = 1, pady = 8)
# L4.pack(side=TOP,pady=5)

E4 = Entry(TAB3)
E4.grid(row = 8, column = 4, pady = 8)
# E4.pack(pady=10)

L5 = Label(TAB3, text="basin_num", font="Times 15 bold")
L5.grid(row = 8, column = 1, pady = 8)
# L5.pack(side=TOP,pady=5)

E5 = Entry(TAB3)
E5.grid(row = 9, column = 4, pady = 8)
# E5.pack(pady=10)

L6 = Label(TAB3, text="Percentage of ppt (to be decreased)", font="Times 15 bold")
L6.grid(row = 9, column = 1, pady = 8)
# L6.pack(side=TOP,pady=5)

E6 = Entry(TAB3)
E6.grid(row = 10, column = 4, pady = 8)
# E6.pack(pady=5)

btn1 = Tk.Button(TAB3, text ='Upload xls Data', font = "Times 10 bold", background = 'red', borderwidth='3',command = lambda: process_file1()) 
btn1.grid(row = 4, column = 4, pady = 8)
# btn1.pack(side = TOP, pady = 5)

btn2 = Tk.Button(TAB3, text ='latlong_limit.mat', font = "Times 10 bold",background = 'red', borderwidth='3',command = lambda:process_file2() )
btn2.grid(row = 13, column = 4, pady = 8)
# btn2.pack(side = TOP, pady = 5)

# button.pack(side=TOP, pady=5)
btn3 = Tk.Button(TAB3, text ='basin_daily_precipitation.mat', font = "Times 10 bold",background = 'red', borderwidth='3',command = lambda: process_file3()) 
btn3.grid(row = 14, column = 4, pady = 8)
# btn3.pack(side = TOP, pady = 5)

btn4 = Tk.Button(TAB3, text ='basin_mat_delineated.mat', font = "Times 10 bold", borderwidth='3',background = 'red',command = lambda: process_file4(E3.get(),E4.get(),E5.get(),E6.get())) 
btn4.grid(row = 15, column = 4, pady = 8)
# btn4.pack(side = TOP, pady = 5)

# button = Button(root, text='Quit', command=_quit)
# btn4.grid(row = 16, column = 4, pady = 8)
# button.pack(side=TOP, pady=5)
L7 = Label(TAB1, text="Shape File to Mask File", font='TkHeadingFont 20 bold')
L7.grid(row = 3, column = 7, pady = 8)

######################

btn5 = Tk.Button(TAB1, text ='Upload Shape File', font = "Times 10 bold",  borderwidth='3',background = 'red',command = lambda: process_file5()) 
btn5.grid(row =4, column = 7, pady = 8)

L8 = Label(TAB1, text="Value of threshold", font="Times 15 bold")
L8.grid(row = 5, column = 7, pady = 8)

E7 = Entry(TAB1)
E7.grid(row = 6, column = 7, pady = 8)

btn6 = Tk.Button(TAB1, text ='Run Part 1', font = "Times 10 bold", borderwidth='3',background = 'red',command = lambda: run_part1(filename5,E7.get()))
btn6.grid(row =7, column = 7, pady = 8)

# L1.pack(side=TOP,pady=5)
# L1.grid(row = 1, column = 1, pady = 4)

# L7 = Label(root, text="Part3-Shape File to Mask File", font='TkHeadingFont 20')
# L7.grid(row = 3, column = 7, pady = 8)

# ######################

# btn5 = Button(root, text ='Upload Shape File', command = lambda: process_file5()) 
# btn5.grid(row =4, column = 7, pady = 8)

# L8 = Label(root, text="Value of threshold", font="Times 15")
# L8.grid(row = 5, column = 7, pady = 8)

# E7 = Entry(root)
# E7.grid(row = 6, column = 7, pady = 8)

# btn6 = Button(root, text ='Run Part 1',command = lambda: run_part1(filename5,E7.get())) 
# btn6.grid(row =7, column = 7, pady = 8)

############################################################
row_start = 2
L9 = Label(TAB4, text="Find Standard Precipitation Index & Standard Runoff Index", font='TkHeadingFont 20 bold')
L9.grid(row = row_start, column = 4, pady = 8)

btn7 = Tk.Button(TAB4, text ='Upload Precipitation file', font = "Times 10 bold",  borderwidth='3',background = 'red',command = lambda: process_file6()) 
btn7.grid(row =row_start + 1, column = 4, pady = 8)

L8 = Label(TAB4, text="Value of time period", font="Times 15 bold")
L8.grid(row = row_start + 2, column = 4, pady = 8)

E8 = Entry(TAB4)
E8.grid(row = row_start + 3, column = 4, pady = 8)

btn8 = Tk.Button(TAB4, text ='Get SPI Data', font = "Times 10 bold", borderwidth='3', background = 'red', command = lambda: run_bonus(filename6,E8.get())) 
btn8.grid(row =row_start + 4, column = 4, pady = 8)

L15 = Label(TAB4, text="SRI Data", font="Times 15 bold")
L15.grid(row = row_start + 5, column = 4, pady = 8)

btn15 = Tk.Button(TAB4, text ='Get SRI Data', font = "Times 10 bold", borderwidth='3', background = 'red', command = lambda: get_SRI(10))
btn15.grid(row =row_start + 6, column = 4, pady = 8)

#############################################################
L9 = Label(TAB2, text="WaterShed Delinitation", font='TkHeadingFont 20 bold')
L9.grid(row = 2, column = 4, pady = 8)

btn9 = Tk.Button(TAB2, text ='Upload DEM Data',  font = "Times 10 bold", borderwidth='3', background = 'red', command = lambda: process_file7())
# btn.pack(side = TOP, pady = 5)
btn9.grid(row = 4, column = 4, pady = 8)
btn10 = Tk.Button(TAB2, text ='Discharge Locations CSV', font = "Times 10 bold", borderwidth='3', background = 'red', command = lambda: process_file8())
# btn.pack(side = TOP, pady = 5)
btn10.grid(row = 5, column = 4, pady = 8)

##########################################################################
# L11 = Label(TAB2, text="Part-2: RiverBasin Delineation", font='TkHeadingFont 20')
# # L0.pack(side=TOP,pady=5)
# L11.grid(row = 6, column = 4, padx = 3, pady = 10)

# def process_file9():
# 	global filename9
# 	file_ = askopenfilename(filetypes =[('Mat Files', '*.mat')])
# 	if file_ is not None:
# 		filename9 = file_
# def process_file10():
# 	global filename10
# 	file_ = askopenfilename(filetypes =[('Mat Files', '*.mat')])
# 	if file_ is not None:
# 		filename10 = file_
# def process_file11():
# 	global filename11
# 	file_ = askopenfilename(filetypes =[('Mat Files', '*.mat')])
# 	if file_ is not None:
# 		filename11 = file_
# 	if filename9 is not None and filename10 is not None and filename11 is not None:
# 		Image.show('delin.png')

btn11 = Button(TAB2, text ='Basin Mat', style = 'NuclearReactor.TButton', command = lambda: process_file9())
# btn.pack(side = TOP, pady = 5)
btn11.grid(row = 7, column = 4, pady = 8)
btn12 = Button(TAB2, text ='Latlong Limit', style = 'NuclearReactor.TButton', command = lambda: process_file10())
# btn.pack(side = TOP, pady = 5)
btn12.grid(row = 8, column = 4, pady = 8)
btn13 = Button(TAB2, text ='Daily Precipitation', style = 'NuclearReactor.TButton', command = lambda: process_file11())
# btn.pack(side = TOP, pady = 5)
btn13.grid(row = 9, column = 4, pady = 8)

L12 = Label(TAB2, text="Part-3: Regression Analysis", font='TkHeadingFont 20')
# L0.pack(side=TOP,pady=5)
L12.grid(row = 7, column = 4, padx = 3, pady = 10)

def process_file12():
	global filename12
	file_ = askopenfilename(filetypes =[('Xls files', '*.xls')])
	if file_ is not None:
		filename12 = file_
def process_file13():
	global filename13
	file_ = askopenfilename(filetypes =[('Mat Files', '*.mat')])
	if file_ is not None:
		filename13 = file_
def process_file14():
	global filename14
	file_ = askopenfilename(filetypes =[('Mat Files', '*.mat')])
	if file_ is not None:
		filename14 = file_
def process_file15():
	global filename15
	file_ = askopenfilename(filetypes =[('Mat Files', '*.mat')])
	if file_ is not None:
		filename15 = file_
	if filename12 is not None and filename13 is not None and filename14 is not None and filename15 is not None:
		import time
		time.sleep(5)
		import matplotlib.image as mpimg
		img=mpimg.imread('vijayawada_ols.png')
		imgplot = plt.imshow(img)


L11 = Label(TAB2, text="Part-2: RiverBasin Delineation", font='TkHeadingFont 20')
# L0.pack(side=TOP,pady=5)
L11.grid(row = 3, column = 4, padx = 3, pady = 10)

def process_file9():
	global filename9
	file_ = askopenfilename(filetypes =[('Mat Files', '*.mat')])
	if file_ is not None:
		filename9 = file_
def process_file10():
	global filename10
	file_ = askopenfilename(filetypes =[('Mat Files', '*.mat')])
	if file_ is not None:
		filename10 = file_
def process_file11():
	global filename11
	file_ = askopenfilename(filetypes =[('Mat Files', '*.mat')])
	if file_ is not None:
		filename11 = file_
	if filename9 is not None and filename10 is not None and filename11 is not None:
		import time
		time.sleep(5)
		import matplotlib.image as mpimg
		img=mpimg.imread('delin.png')
		imgplot = plt.imshow(img)
btn11 = Button(TAB2, text ='Basin Mat', style = 'NuclearReactor.TButton', command = lambda: process_file9())
# btn.pack(side = TOP, pady = 5)
btn11.grid(row = 4, column = 4, pady = 8)
btn12 = Button(TAB2, text ='Latlong Limit', style = 'NuclearReactor.TButton', command = lambda: process_file10())
# btn.pack(side = TOP, pady = 5)
btn12.grid(row = 5, column = 4, pady = 8)
btn13 = Button(TAB2, text ='Daily Precipitation', style = 'NuclearReactor.TButton', command = lambda: process_file11())
# btn.pack(side = TOP, pady = 5)
btn13.grid(row = 6, column = 4, pady = 8)








Tk.mainloop()

