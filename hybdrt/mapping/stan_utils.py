from cmdstanpy import CmdStanModel
from pathlib import Path
import pickle

_module_dir = Path(__file__).parent

def load_model(name):
    if name[-4:] != '.stan':
        name = f'{name}.stan'
        
    model = CmdStanModel(
        stan_file=_module_dir.joinpath('stan_models', name)
        )
    
    return model


def save_pickle(obj, dest):
    with open(dest, 'wb') as f:
        pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)
        
        
def load_pickle(src):
    with open(src, 'rb') as f:
        out = pickle.load(f)
    return out
