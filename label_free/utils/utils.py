

from dataclasses import dataclass
import importlib
from typing import Dict



def load_object(name:str, kwargs:Dict):
    """
    Instantiates object from object name and kwargs
    """
    object_module, object_name = name.rsplit(".", 1)
    object_module = importlib.import_module(object_module)


    return getattr(object_module, object_name)(**kwargs)