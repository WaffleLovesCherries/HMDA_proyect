import os
from pickle import load, dump
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from time import time

with open('model_list.txt', 'r') as file:
    model_filenames = file.read().splitlines()

with open('data.pkl','rb') as file:
    X, y = load( file )

X, y = X.copy(), y.copy()

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

if __name__ == '__main__':

    models = []

    for model_name in model_filenames:
        model_path = os.path.join('models', f'{model_name}.pkl')
        with open( model_path, 'rb' ) as f:
            pipe, param_grid = load( f )

        model = BayesSearchCV( estimator = pipe, search_spaces = param_grid, cv = cv, n_iter=30, n_jobs=5, scoring='roc_auc', verbose=5 )
        
        print(f'Training model {model_name}.')
        start = time()
        model.fit( X, y )
        end = time()
        print(f'Model trained with {model.best_params_}.')

        with open( f'model_trained/{model_name}_trained.pkl', 'wb' ) as f:
            dump( {'model': model, 'cpu_time': end - start, 'name': model_name}, f )



