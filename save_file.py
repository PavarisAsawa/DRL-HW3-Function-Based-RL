import os
import json

def save_hyperparam(experiment_path, hyperparam):
    '''
        Input with Dictionary of hyperparameter
    '''
    os.makedirs(experiment_path, exist_ok=True)
    # Save the JSON file
    with open(os.path.join(experiment_path, "hyperparam.json"), "w") as f:
        json.dump(hyperparam, f, indent=4)