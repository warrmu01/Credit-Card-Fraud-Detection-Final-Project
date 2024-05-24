# Similar to creditcard_alg_investigation.ipynb (with various changes for use in Project 2)


import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


from creditcard_preparation import create_creditcard_pipeline, prepare_creditcard_data


print("Preparing Project 1 Models")

# Functions to evaluate an algorithm

def evaluate_algo(algo, X_train, y_train, X_test, y_test):
    # Create the pipeline

    pipeline = create_creditcard_pipeline()

    # Combine the pipeline and the algorithm
    pipeline_with_algo = Pipeline(steps=[
        ('preprocessor', pipeline),
        ('algo', algo)
    ])

    pipeline_with_algo.fit(X_train, y_train)
    y_pred = pipeline_with_algo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Function for LogisticRegression


def evaluate_lr(X_train, y_train, X_test, y_test):
    print("Evaluating LogisticRegression...")
    return evaluate_algo(LogisticRegression(max_iter=1000, random_state=42), X_train, y_train, X_test, y_test)

# Function for SVC


def evaluate_svc(X_train, y_train, X_test, y_test):
    print("Evaluating SVC...")
    return evaluate_algo(SVC(random_state=42), X_train, y_train, X_test, y_test)

# Function for KNeighborsClassifier


def evaluate_knn(X_train, y_train, X_test, y_test):
    print("Evaluating KNeighborsClassifier...")
    return evaluate_algo(KNeighborsClassifier(), X_train, y_train, X_test, y_test)

# Function for DecisionTreeClassifier


def evaluate_dt(X_train, y_train, X_test, y_test):
    print("Evaluating DecisionTreeClassifier...")
    return evaluate_algo(DecisionTreeClassifier(random_state=42), X_train, y_train, X_test, y_test)




# Prepare credit card data for train

X_train, X_dev, X_test, y_train, y_dev, y_test = prepare_creditcard_data(ratios=((1/10), (1/10)))



# Evaluate algorithms
lr_scores = evaluate_lr(X_train, y_train, X_test, y_test)
svc_scores = evaluate_svc(X_train, y_train, X_test, y_test)
knn_scores = evaluate_knn(X_train, y_train, X_test, y_test)
dt_scores = evaluate_dt(X_train, y_train, X_test, y_test)

print()

# Create DataFrame to store scores
scores_df = pd.DataFrame([lr_scores, svc_scores, knn_scores, dt_scores],
                         columns=['Accuracy'],
                         index=['LogisticRegression', 'SVC', 'KNeighborsClassifier',
                                'DecisionTreeClassifier'])
