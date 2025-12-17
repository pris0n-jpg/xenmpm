from pathlib import Path
PROJ_DIR = Path(__file__).resolve().parent

import os
from distutils.util import strtobool

def getEnvBool(env_var, default=False) -> bool:
    return strtobool(os.environ.get(env_var, str(default)))

def getEnvStr(env_var, default='') -> str:
    return os.environ.get(env_var, default)