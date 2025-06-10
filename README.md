Malicious Activity Detection with MLP

Project Overview

This project implements a Multi-Layer Perceptron (MLP) classifier to detect malicious activities in a cybersecurity dataset, involving data preprocessing, feature selection, hyperparameter tuning via grid search, and performance evaluation using accuracy, precision, recall, and confusion matrices. The goal is to develop an optimized neural network model that effectively identifies malicious behavior while providing insights into its performance and limitations.

Features





Data Preprocessing: Encodes categorical variables, scales features using Min-Max scaling, and removes duplicates to ensure a clean dataset.



Feature Selection: Utilizes correlation analysis with a heatmap to select relevant features, reducing dimensionality and improving model efficiency.



MLP Classifier: Trains a neural network with configurable hidden layers, activation functions, and solvers for robust classification.



Hyperparameter Tuning: Employs grid search with cross-validation to optimize parameters like hidden layer sizes, learning rates, and regularization.



Performance Evaluation: Analyzes model performance with accuracy, precision, recall, and confusion matrix visualizations.

Project Structure

Malicious-Activity-Detection-MLP/

├── cleaning.py               # Data preprocessing and cleaning script

├── Feature_Selection.py      # Feature selection using correlation heatmap

├── Model-Training.py         # MLP classifier training and evaluation

├── Tuning.py                 # Hyperparameter tuning with grid search

├── preprocessed_TrainData.csv # Preprocessed dataset

├── README.md                 # Project documentation


Installation & Setup Instructions





Clone the Repository:
git clone https://github.com/m-abdullah3/Malicious-Activity-Detection-MLP.git
cd Malicious-Activity-Detection-MLP
Set Up a Python Environment:





Ensure Python 3.8+ is installed.



Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
pip install pandas scikit-learn matplotlib seaborn

Prepare the Dataset:





Place the preprocessed_TrainData.csv file in the project directory.

Run the Project Scripts:





Perform feature selection:

python Feature_Selection.py



Train and evaluate the MLP model:

python Model-Training.py



Tune hyperparameters:

python Tuning.py

Tech Stack / Libraries Used





Python: Core programming language.



Pandas: Data manipulation and preprocessing.



Scikit-learn: MLP classifier, grid search, and performance metrics.



Matplotlib: Visualization of confusion matrices and histograms.



Seaborn: Heatmap for feature correlation analysis.

License

This project is licensed under the MIT License - see the LICENSE file for details.



