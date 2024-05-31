"""Utils for sonify."""

from importlib.resources import files
from functools import partial
from dol import Pipe
from config2py import (
    process_path,
    simple_config_getter,
)

pkg_name = 'sonify'

data_files = files(pkg_name) / 'data'

get_config = simple_config_getter(pkg_name)

