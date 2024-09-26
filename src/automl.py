import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from preprocess import preprocess_data

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def objective(params):
    model = RandomForestClassifier(**params)
    X_train, X_test, y_train, y_test = preprocess_data(load_data('data/sample_dataset.csv'))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {'loss': -accuracy_score(y_test, preds), 'status': STATUS_OK}

if __name__ == "__main__":
    space = {
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'max_depth': hp.choice('max_depth', [None, 10, 20, 30]),
        'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.1),
    }
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50)
    print("Best parameters:", best)

