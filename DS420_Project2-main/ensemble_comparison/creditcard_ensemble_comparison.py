from .creditcard_ensemble_bag import best_model as bag_model # 30 seconds to 1 minute, 30 seconds
from .creditcard_ensemble_RFC import best_model as RFC_model # 30 seconds to 1 minute
from .creditcard_ensemble_AB import best_model as AB_model # 6-10 minutes
from .creditcard_ensemble_GB import best_model as GB_model # 30 minutes
from .creditcard_ensemble_HGB import best_model as HGB_model # 30 seconds
from .creditcard_ensemble_SC import best_model as stack_model # 20 minutes
from .creditcard_ensemble_voting import best_model as voting_model # 10 minutes


import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score


from creditcard_preparation import prepare_creditcard_data, combine_algo_and_pipeline



def new_eval(algo, X_train, y_train, X_test, y_test):

    try:

        algo.fit(X_train,np.ravel(y_train))

        test_predictions = algo.predict(X_test)

        accuracy = accuracy_score(y_test, test_predictions)

        scores = {"Accuracy": accuracy}

    except:

        # alternate data (no raveling performed)

        algo.fit(X_train,y_train)

        test_predictions = algo.predict(X_test)
    
        accuracy = accuracy_score(y_test, test_predictions)

        scores = {"Accuracy": accuracy}

        

    return scores



print()
print("Preparing Ensemble Models")


# Run all models and combine results into dataframe

X_train, X_dev, X_test, y_train, y_dev, y_test = prepare_creditcard_data((1/10,1/10))

print("Data Loaded")

models = [bag_model,
          RFC_model,
          AB_model,
          GB_model,
          HGB_model,
          voting_model,
          stack_model]


data = []

for model in models:

    print("Testing the", model)

    model_prepared = combine_algo_and_pipeline(model)

    data.append(new_eval(model_prepared, X_train, y_train, X_test, y_test))



scores = pd.DataFrame(data=tuple(data),
                     index=["BaggingClassifier",
                            "RandomForestClassifier",
                            "AdaBoostClassifier",
                            "GradientBoostingClassifier",
                            "HistGradientBoostingClassifier",
                            "VotingClassifier",
                            "StackingClassifier"]
                     ).sort_values("Accuracy", ascending=False)