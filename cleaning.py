import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Loading the dataset as a pandas frame
dataFrame= pd.read_csv("Train_data.csv")
print ("shape = ",dataFrame.shape)

#Printing 1st five rows using head()
print("Head: ")
print(dataFrame.head().to_string())
print("------------------------------------------------")

#Printing last five rows using tail()
print("Tail")
print(dataFrame.tail().to_string())
print("------------------------------------------------")

#Finding Missing Values
print("Missing Values")
print(dataFrame.isnull().sum())
print("------------------------------------------------")

#Finding Duplicate values
print("number of Duplicates = ",dataFrame.duplicated().sum())

#Removing duplicates
dataFrame.drop_duplicates(inplace=True)
print("Removing Duplictes..........")
print("number of Duplicates = ", dataFrame.duplicated().sum())

print("------------------------------------------------")

#getting statistics about the data using describe
print("Describe()")
print(dataFrame.describe().to_string())

print("------------------------------------------------")

#Getting information about the dataframe using info()
print("Info()")
print(dataFrame.info())

print("------------------------------------------------")

#Checking count of unique labels
print("Count of Labels")
labelCount=dataFrame["class"].value_counts()
#printing the result
print(labelCount)
print("------------------------------------------------")

#Encoding non-numeric values to numeric
labelEncoder=preprocessing.LabelEncoder()
dataFrame["protocol_type"]=labelEncoder.fit_transform(dataFrame["protocol_type"])
dataFrame["service"]=labelEncoder.fit_transform(dataFrame["service"])
dataFrame["flag"]=labelEncoder.fit_transform(dataFrame["flag"])
dataFrame["class"]=labelEncoder.fit_transform(dataFrame["class"])



#Plotting a histogram to for visual analysis
plt.hist(dataFrame, bins=1, edgecolor='black')
#setting title
plt.title("Histogram")
plt.show()

#defining the Min-Max feature scaler
featuresScaler=preprocessing.MinMaxScaler()
#getting the columns that require scaling
features=dataFrame.drop("class", axis=1).columns

#Scaling the features
scaledFeatures = featuresScaler.fit_transform(dataFrame[features])
#Converting the scaled features into a data frame
scaledDataframe = pd.DataFrame(scaledFeatures, columns=features)

# Adding the labels to the scaled dataframe
scaledDataframe["class"] = dataFrame["class"].values

print("------------------------------------------------")
print('Cleaned Dataset')
#printed the head of the cleaned dataset
print(scaledDataframe.head())

#saving the cleaned dataset to a csv file
cleanedDataset ="preprocessed_TrainData.csv"
scaledDataframe.to_csv(cleanedDataset,index=False)
