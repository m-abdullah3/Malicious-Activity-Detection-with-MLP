from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


#Loading the dataset as a pandas frame
dataFrame=pd.read_csv("preprocessed_TrainData.csv")

#Printing 1st five rows
print(dataFrame.head())

#seperating the features and the labels
X=dataFrame.drop("class",axis=1)
Y=dataFrame["class"]
#spliting the dataset into 10% validation set
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.10)

# Define the MLPClassifier
model = MLPClassifier(max_iter=100)

# Define the grid of hyperparameters
parameterToBeTuned = {
    'hidden_layer_sizes': [(200,), (500,), (200, 200)],
    'activation': ['relu', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive',],
    'learning_rate_init': [0.001, 0.01, 0.1],
}

# Perform grid search
gridSearch= GridSearchCV(model, parameterToBeTuned, cv=3,verbose=2)


#Fitting the grid search to find the best result through cross validation
#k=3 folds is set
gridSearch.fit(X_train, y_train)

# Print best parameters and score
print('Best parameters found:\n', gridSearch.best_params_)

