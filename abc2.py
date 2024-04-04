import numpy as np
import pandas as pd
from sklearn import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
#Read the dataset into a dataframe
df = pd.read_csv('taxset.csv')


#Mapping of Sales Vs.Purchases
df["SalesVsPurchases"] = df["SalesVsPurchases"].map({'SVgtPV':3 ,'SVeqPV':2,'SVltPV':1})


#Mapping of Employee Growth
df["EmployeeGrowth"] = df["EmployeeGrowth"].map({'EGincrTPdecr':3 ,'EGincrTPincr':2,'EGdecrTPincr':1})


#Mapping of Total Sales
df["TotalSales"] = df["TotalSales"].map({'TSincrTPdecr':3 ,'TSincrTPincr':2,'TSdecrTPincr':1})


#Mapping of Claim Credits
df["ClaimCredits"] = df["ClaimCredits"].map({'UnusualCC':2,'UsualCC':1})


#Mapping of Net Income
df["NetIncome"] = df["NetIncome"].map({'NIincrTIdecr':3 ,'NIincrTIincr':2,'NIdecrTIincr':1})


#Mapping of Gross Loss
df["GrossLoss"] = df["GrossLoss"].map({'GLdecrTaxdecr':3 ,'GLincrTaxdecr':2,'GLdecrTaxincr':1})


#Mapping of Inventories
df["Inventories"] = df["Inventories"].map({'INVincrDIdecr':3 ,'INVincrDIincr':2,'INVdecrDIincr':1})


#Mapping of Gross Revenue
df["GrossRevenue"] = df["GrossRevenue"].map({'GRincrTPdecr':3 ,'GRincrTPincr':2,'GRdecrTPincr':1})


#Mapping of Raw Materials
df["RawMaterials"] = df["RawMaterials"].map({'RMPCincrFPSdecr':3 ,'RMPCincrFPSincr':2,'RMPCdecrFPSincr':1})


#Mapping of Fixed Assets
df["FixedAssets"] = df["FixedAssets"].map({'FAincrTPdecr':3 ,'FAincrTPincr':2,'FAdecrTPincr':1})

#Mapping of Tax Evasion
df["TaxEvasion"] = df["TaxEvasion"].map({'High':2 ,'Medium':1,'Low':0})


#Creation of data as numpy array
data = df[["SalesVsPurchases","EmployeeGrowth","TotalSales","ClaimCredits","NetIncome","GrossLoss","Inventories","GrossRevenue","RawMaterials","FixedAssets","TaxEvasion"]].to_numpy()


#All columns except last column are considered as inputs
inputs = data[:,:-1]
outputs = data[:, -1]
training_inputs = inputs[:1000]
training_outputs = outputs[:1000]
testing_inputs = inputs[1000:]
testing_outputs = outputs[1000:]
classifier = AdaBoostClassifier()
classifier.fit(training_inputs, training_outputs)
testSet = [[3,3,2,1,3,2,2,2,1,2]]
test = pd.DataFrame(testSet)
predictions = classifier.predict(test)
print('ETC prediction on the first test set is:',predictions)
testSet = [[2,1,1,2,3,3,2,3,2,3]]
test = pd.DataFrame(testSet)
predictions = classifier.predict(test)
print('ETC prediction on the second test set is:',predictions)
testSet = [[1,2,2,1,2,1,2,1,1,3]]
test = pd.DataFrame(testSet)
predictions = classifier.predict(test)
print('ETC prediction on the third test set is:',predictions)