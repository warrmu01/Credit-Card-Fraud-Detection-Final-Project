from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


best_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 10, 
                                    min_samples_leaf=10,
                                    random_state=42, max_features = 0.8), 
                                 n_estimators = 50, learning_rate = 1.0)