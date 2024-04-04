#Importing Numpy
import numpy as np


#Importing Pandas
import pandas as pd


#Importing Matpplotlib
import matplotlib.pyplot as plt



#Importing Tensorflow
import tensorflow as tf


#Importing Keras backend
import tensorflow.keras.backend as K



#Print Tensorflow Version
print(tf.__version__)



#Importing Warnings
import warnings

#Ignoring Warnings
warnings.filterwarnings("ignore")


#Function for getting F1-Score
def get_precision(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return precision


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


#Last Column is considered as outputs
outputs = data[:, -1]



#First Thousand rows are considered for training.
training_data = inputs[:600]


#Training labels are set to the last column values of first thousand rows
training_labels = outputs[:600]



#Remaining Rows, Beyond 600 are considered for testing
test_data = inputs[600:]


#Testing labels are set to the last column values of remaining rows
test_labels = outputs[600:]


#Tensorflow Initiation
tf.keras.backend.clear_session()




#Configure the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(32, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
									
									
#Comiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



#Creation of the model
model.fit(training_data, training_labels, epochs=100)


#First Test Set Assinment
testSet = [[3,3,2,1,3,2,2,2,1,2]]


#First Test Set Conversion to Pandas Data Frame
test = pd.DataFrame(testSet)


#Prediction on First Test Set Using the Model
predictions = model.predict(test)


#Finding the first test set label
classes=np.argmax(predictions,axis=1)


#printing the first test set label
print('DL Model Prediction on the first test set is:',classes)


#Second Test Set Assinment
testSet = [[2,1,1,2,3,3,2,3,2,3]]


#Second Test Set Conversion to Pandas Data Frame
test = pd.DataFrame(testSet)


#Prediction on Second Test Set Using the Model
predictions =  model.predict(test)


#Finding the second test set label
classes=np.argmax(predictions,axis=1)


#printing the second test set label
print('DL Model Prediction on the second test set is:',classes)

#Third Test Set Assinment
testSet = [[1,2,2,1,2,1,2,1,1,3]]


#Third Test Set Conversion to Pandas Data Frame
test = pd.DataFrame(testSet)


#Prediction on Third Test Set Using the Model
predictions =  model.predict(test)


#Finding the Third test set label
classes=np.argmax(predictions,axis=1)


#printing the Third test set label
print('DL Model Prediction on the Third test set is:',classes)