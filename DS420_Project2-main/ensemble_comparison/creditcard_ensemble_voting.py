from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier


best_model = VotingClassifier(n_jobs = -1, estimators=
            [('dt', DecisionTreeClassifier()),
            ('lr', LogisticRegression()),
            ('rfc', RandomForestClassifier())])