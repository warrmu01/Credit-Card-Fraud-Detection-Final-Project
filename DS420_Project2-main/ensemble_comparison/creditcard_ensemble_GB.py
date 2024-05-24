from sklearn.ensemble import GradientBoostingClassifier


best_model =  GradientBoostingClassifier(max_depth = 10, 
                                    min_samples_leaf=10,
                                    random_state=42, max_features = 0.8, n_iter_no_change=2,
                                    n_estimators = 500, learning_rate = 0.1,
                                    subsample = 0.9, loss="log_loss")