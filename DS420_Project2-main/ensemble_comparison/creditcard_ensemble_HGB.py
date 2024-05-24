import numpy as np


from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from creditcard_preparation import prepare_creditcard_data, combine_algo_and_pipeline








def main():
    X_train, X_dev, X_test, y_train, y_dev, y_test = prepare_creditcard_data((1/10,1/10))


    #
    # ---
    #



    # Histogram Gradient Boosting

    HGBClassifier = combine_algo_and_pipeline(HistGradientBoostingClassifier())



    HGBClassifier.fit(X_train,np.ravel(y_train))


    HGBtest_predictions = HGBClassifier.predict(X_test)
    HGBtrain_predictions = HGBClassifier.predict(X_train)

    HGBtest_rmse = mean_squared_error(y_test, HGBtest_predictions)
    HGBtrain_rmse = mean_squared_error(y_train, HGBtrain_predictions)

    print(X_train.shape)
    print("HistogramGradienttBoostingClassifier: ")
    print("train, test error rates:", HGBtrain_rmse / np.mean(y_train),#.values,
                                    HGBtest_rmse / np.mean(y_test))#.values)


    #
    # ---
    #

    # Hypertune HistGradientBoostingClassifier

    param_grid = {
    'algo__learning_rate': [0.01, 0.1, 1.0],
    'algo__l2_regularization': [0.0, 0.1, 0.2],
    }

    # Create a HistGradientBoostingClassifier
    HGBClassifier = combine_algo_and_pipeline(HistGradientBoostingClassifier())

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=HGBClassifier, param_grid=param_grid, scoring='accuracy', cv=2, verbose=3)
    grid_search.fit(X_train, y_train)

    # Retrieve the best hyperparameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Make predictions on the testing data
    test_predictions = best_model.predict(X_test)
    train_predictions = best_model.predict(X_train)

    # Evaluate the performance
    test_rmse = mean_squared_error(y_test, test_predictions)
    train_rmse = mean_squared_error(y_train, train_predictions)

    print(X_train.shape)
    print("HistGradientBoostingClassifier: ")
    print(best_params)
    print("train, test error rates:", train_rmse / np.mean(y_train),#.values,
                                    test_rmse / np.mean(y_test))#.values)


    #
    # ---
    #

best_model = HistGradientBoostingClassifier(
    l2_regularization=0.2,
    learning_rate=1.0
)


if __name__ == "__main__":
    main()

