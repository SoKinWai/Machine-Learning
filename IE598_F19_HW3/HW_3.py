# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 10:48:24 2019

@author: Jianwei Su
"""

#Listing 2-1
import numpy as np
import sys
#read data from my laptop
with open('D:\Machine Learning part 1\HW_3\HY_Universe_corporate bond.csv') as data:
#arrange data into list for labels and list of lists for attributes
    xList = []
    labels = []
    next(data)
    for line in data:
    #split on comma
        row = line.strip().split(",")
        xList.append(row)
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1])))

#Listing 2-2
print("\n")
nrow = len(xList)
ncol = len(xList[1])
type = [0]*3
colCounts = []
for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    colCounts.append(type)
    type = [0]*3
sys.stdout.write("Col#" + '\t' + "Number" + '\t' + "Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
    str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1

#Listing 2-3
#we can also use col=9.
col = 10
colData = []
for row in xList:
    colData.append(float(row[col]))
colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' + "Standard Deviation = " + '\t ' + str(colsd) + "\n")
#calculate quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")
#run again with 10 equal intervals
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")
#The last column contains categorical variables
col = 29
colData = []
for row in xList:
    colData.append(row[col])
unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)
#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*5
for elt in colData:
    catCount[catDict[elt]] += 1
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)

#Listing 2-4
from matplotlib import pylab
import scipy.stats as stats
#generate summary statistics for column 15 (e.g.)
col = 15
colData = []
for row in xList:
    colData.append(float(row[col]))
stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()

#Listing 2-5
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
#read HY data into pandas data frame
hy = pd.read_csv('D:\Machine Learning part 1\HW_3\HY_Universe_corporate bond.csv')
#print head and tail of data frame
print(hy.head())
print(hy.tail())
#print summary of data frame
summary = hy.describe()
print(summary)

#Listing 2-6
hy_drop = hy.drop(['CUSIP','Ticker','Issue Date','Maturity','1st Call Date','Moodys','S_and_P',
              'Fitch','Bloomberg Composite Rating','Maturity Type','Coupon Type','Industry',
              'Months in JNK','Months in HYG','Months in Both','IN_ETF'],axis=1)
print(hy_drop.head())
cols = list(hy_drop)
pcolor = []
for row in range(nrow):
    if hy_drop.iat[row,20] == 1:
        pcolor = "blue"
    elif hy_drop.iat[row,20] == 2:
        pcolor = "red"
    elif hy_drop.iat[row,20] == 3:
        pcolor = "purple"
    elif hy_drop.iat[row, 20] == 4:
        pcolor = "yellow"
    elif hy_drop.iat[row, 20] == 5:
        pcolor = "orange"
    # plot rows of data as if they were series data
    dataRow = hy_drop.iloc[row, 0:20]
    dataRow.plot(color=pcolor, alpha=0.5)
plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()

#Listing 2-7
#calculate correlations between real-valued attributes
dataRow36 = hy.iloc[:,35]
dataRow14 = hy.iloc[:,13]
plot.scatter(dataRow36, dataRow14)
plot.xlabel("34th Attribute")
plot.ylabel(("12th Attribute"))
plot.show()

dataRow16 = hy.iloc[:,15]

plot.scatter(dataRow36, dataRow16)
plot.xlabel("36th Attribute")
plot.ylabel(("16th Attribute"))
plot.show()

#Listing 2-8
#change the targets to numeric values
from random import uniform
target = []
for row in range(nrow):
    if hy.iat[row,29] == 1:
        target.append(1.0)
    elif hy.iat[row,29] == 2:
        target.append(2.0)
    elif hy.iat[row,29] == 3:
        target.append(3.0)
    elif hy.iat[row, 29] == 4:
        target.append(4.0)
    elif hy.iat[row, 29] == 5:
        target.append(5.0)
dataRow = hy.iloc[0:nrow,35]
plot.scatter(dataRow,target)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()

target = []
for row in range(nrow):
    if hy.iat[row,29] == 1:
        target.append(1.0+ uniform(-0.1, 0.1))
    elif hy.iat[row,29] == 2:
        target.append(2.0+ uniform(-0.1, 0.1))
    elif hy.iat[row,29] == 3:
        target.append(3.0+ uniform(-0.1, 0.1))
    elif hy.iat[row, 29] == 4:
        target.append(4.0+ uniform(-0.1, 0.1))
    elif hy.iat[row, 29] == 5:
        target.append(5.0+ uniform(-0.1, 0.1))
dataRow = hy.iloc[0:nrow,35]
plot.scatter(dataRow,target, alpha=0.5, s=120)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()

#Listing 2-9
from math import sqrt
mean36=0.0;
mean14=0.0;
mean16=0.0
numElt = len(dataRow14)
for i in range(numElt):
    mean36 += dataRow36[i]/numElt
    mean14 += dataRow14[i]/numElt
    mean16 += dataRow16[i]/numElt
var36 = 0.0; var14 = 0.0; var16 = 0.0
for i in range(numElt):
    var14+= (dataRow14[i] - mean14) * (dataRow14[i] - mean14)/numElt
    var16 += (dataRow16[i] - mean16) * (dataRow16[i] - mean16)/numElt
    var36 += (dataRow36[i] - mean36) * (dataRow36[i] - mean36)/numElt
corr1416 = 0.0; corr1436 = 0.0
for i in range(numElt):
  corr1416 += (dataRow14[i] - mean14) * \
              (dataRow16[i] - mean16) / (sqrt(var14*var16) * numElt)
              
  corr1436 += (dataRow14[i] - mean14) * \
               (dataRow36[i] - mean36) / (sqrt(var14*var36) * numElt)            
sys.stdout.write("Correlation between attribute 14 and 16 \n")
print(corr1416)
sys.stdout.write(" \n")

sys.stdout.write("Correlation between attribute 14 and 36 \n")
print(corr1436)
sys.stdout.write(" \n")

#Listing 2-10
#calculate correlations between real-valued attributes
corMat = DataFrame(hy.corr())
#visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()

print("My name is Jianwei Su")
print("My NetID is: jianwei5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")









