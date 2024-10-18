"""Utils for sonification."""

from importlib.resources import files
from functools import partial
from dol import Pipe
from config2py import (
    process_path,
    simple_config_getter,
)
from tonal import DFLT_SOUNDFONT

pkg_name = 'sonification'

data_files = files(pkg_name) / 'data'

get_config = simple_config_getter(pkg_name)


DFLT_OUTPUT_NAME = 'audio_output'
DFLT_MIDI_OUTPUT = f"{DFLT_OUTPUT_NAME}.mid"
DFLT_WAV_OUTPUT = f"{DFLT_OUTPUT_NAME}.wav"
