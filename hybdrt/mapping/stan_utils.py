from cmdstanpy import CmdStanModel
from pathlib import Path


_module_dir = Path(__file__).parent

def load_model(name):
    if name[-4:] != '.stan':
        name = f'{name}.stan'
        
    model = CmdStanModel(
        stan_file=_module_dir.joinpath('stan_models', name)
        )
    
    return model
