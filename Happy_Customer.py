import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Load data from csv file
df = pd.read_csv('ACME-HappinessSurvey2020.csv')

# Split the data into input (X) and output (Y)
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
Y = df['Y']

# Handle imbalance data using SMOTE
sm = SMOTE(random_state=1)
X_res, Y_res = sm.fit_resample(X, Y)

# Split data into training set and test set with stratification
X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.3, random_state=1, stratify=Y_res)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(random_state=1)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Max depth of the trees
    'min_samples_split': [2, 5, 10]  # Minimum number of samples required to split an internal node
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy')

# Fit GridSearchCV to the training data
grid_search.fit(X_train, Y_train)

# Get the best parameters
best_params = grid_search.best_params_
print('Best Parameters:', best_params)

# Train the RandomForestClassifier with the best parameters
clf_best = RandomForestClassifier(**best_params, random_state=1)
clf_best.fit(X_train, Y_train)

# Predict the target for the test data
Y_pred = clf_best.predict(X_test)

# Print the accuracy of the model 
print("Accuracy:", accuracy_score(Y_test, Y_pred))

# Predict the target for a new data point
new_data = pd.DataFrame([[2, 4, 3, 4, 5, 4]], columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6'])
print("Prediction for new data: ", clf_best.predict(new_data)[0])
