from sklearn.ensemble import RandomForestClassifier


best_model = RandomForestClassifier(
    n_estimators=10,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=4,
)