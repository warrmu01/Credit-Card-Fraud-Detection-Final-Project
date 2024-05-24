from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


best_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),
    max_features=0.7,
    max_samples=0.7
)