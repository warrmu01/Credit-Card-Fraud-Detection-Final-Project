from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression


from .creditcard_ensemble_RFC import best_model as best_RFC



best_DTC = DecisionTreeClassifier(criterion="entropy")


best_model = StackingClassifier(estimators=[
    ("RF", best_RFC),
    ("LR", LogisticRegression())
],
final_estimator=best_DTC,
cv= 5, n_jobs=-1)