'''
Data Sonification with Pyo (Fixed, Dynamic Frequency)

This module converts multidimensional data into sound parameters,
creating a sonification of the data attributes. It records each segment
to an in-memory table and returns the concatenated waveform.

Example usage:
    # Assuming score_df already loaded
    waveform = create_sonification(score_df)
    import soundfile as sf
    sf.write("sonification.wav", waveform, 44100)
'''
from typing import Mapping, Sequence, Tuple
import time

import numpy as np

from sonification.pyo_util import Server

from pyo import (
    NewTable, TableRec, Mix,
    Adsr, Phasor, Biquadx, Sine, Disto, WGVerb, PyoObject
)

def scale_value(
    value: float,
    source_range: Tuple[float, float],
    target_range: Tuple[float, float],
) -> float:
    """
    Scale a value from source range to target range.
    """
    src_min, src_max = source_range
    tgt_min, tgt_max = target_range
    if src_max == src_min:
        normalized = 0.5
    else:
        normalized = (value - src_min) / (src_max - src_min)
    return float(tgt_min + normalized * (tgt_max - tgt_min))


def _compute_group_value(row: Mapping, attributes: Sequence[str]) -> float:
    """
    Compute the combined value for a group of attributes.
    """
    return float(np.mean([row[attr] for attr in attributes]))


class Osc303(PyoObject):
    """Basic emulation of a TB-303"""

    def __init__(
        self,
        freq=20.0,
        decay=0.5,
        shape=1.0,
        cutoff=2000.0,
        reso=5.0,
        mul=None,
    ):
        self._freq = float(freq)
        self._shape = float(shape)
        self._cutoff = float(cutoff)
        self._reso = float(reso)
        self._decay = float(decay)

        self._mul = mul if mul is not None else 1.0

        self.wave1 = Phasor(freq=self._freq, phase=0.0, mul=1.0)
        self.wave2 = Phasor(freq=-self._freq, phase=0.5, mul=1.0)
        self.env = Adsr(
            attack=0.05,
            decay=self._decay,
            sustain=0.1,
            release=0.05,
            dur=0.2 + self._decay,
            mul=self._mul,
        )

        shape_wave0 = Phasor(freq=self._freq, phase=0.0, mul=self.env)
        shape_wave1 = ((self.wave1 + self.wave2) - 1.0) * self.env
        self.wave3 = shape_wave0 * (1.0 - self._shape) + shape_wave1 * self._shape

        self.filter = Biquadx(
            [self.wave3, self.wave3],
            freq=self._cutoff,
            q=self._reso,
            type=0,
            stages=2,
        )
        self._base_objs = self.filter.getBaseObjects()

    def play(self):
        self.env.play()
        return self.filter.out()

    def stop(self):
        self.filter.stop()
        return self.env.stop()

    def setFreq(self, x):
        # Accept numeric or PyoObject for dynamic modulation
        self._freq = x
        self.wave1.freq = x
        self.wave2.freq = -x

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, x):
        self.setFreq(x)


def create_sonification(data_df, duration_per_row=0.5, sr=44100) -> np.ndarray:
    """
    Create a sonification of the data and return a waveform array.
    """
    total_duration = duration_per_row * len(data_df)

    server = Server(audio="portaudio", sr=int(sr), nchnls=1, buffersize=512).boot()
    server.start()

    attribute_groups = {
        "positive_emotion": ["emotion_hope", "emotion_pride"],
        "negative_emotion": ["emotion_anger", "emotion_fear", "emotion_despair"],
        "sentiment": ["sentiment_polarity", "moral_outrage"],
        "provocation": ["intent_provocation", "hostility"],
        "style": ["style_urgency", "style_informality", "assertion_strength"],
        "topic": ["topic_security", "topic_resource", "military_intensity"],
    }
    param_mappings = {
        "positive_emotion": {"frequency": (300.0, 800.0), "harmonic_content": (0.1, 0.4), "attack": (0.1, 0.3), "decay": (0.5, 1.5)},
        "negative_emotion": {"frequency": (100.0, 500.0), "harmonic_content": (0.4, 0.9), "attack": (0.01, 0.1), "decay": (0.3, 0.8)},
        "sentiment": {"frequency_mod": (0.0, 8.0), "frequency_mod_depth": (0.0, 0.1)},
        "provocation": {"amplitude": (0.3, 0.8), "distortion": (0.0, 0.7)},
        "style": {"tempo": (1.0, 8.0), "resonance": (0.1, 5.0)},
        "topic": {"filter_freq": (500.0, 5000.0), "reverb": (0.0, 0.5)},
    }

    all_segments = []
    for row in data_df.to_dict(orient='records'):
        oscillators, modulators, filters = {}, {}, {}
        for group, attrs in attribute_groups.items():
            val = _compute_group_value(row, attrs)
            p = param_mappings[group]

            if group in ["positive_emotion", "negative_emotion"]:
                freq = scale_value(val, (0.0, 1.0), p["frequency"])
                shape = scale_value(val, (0.0, 1.0), p["harmonic_content"])
                attack = scale_value(val, (0.0, 1.0), p["attack"])
                decay = scale_value(val, (0.0, 1.0), p["decay"])
                env = Adsr(attack=attack, decay=decay, sustain=0.3, release=0.5, dur=duration_per_row, mul=0.5)
                osc = Osc303(freq=freq, decay=decay, shape=shape, cutoff=3000.0, reso=1.0, mul=env)
                oscillators[group] = osc

            if group == "sentiment" and "positive_emotion" in oscillators:
                rate = scale_value(val, (0.0, 1.0), p["frequency_mod"])
                depth = scale_value(val, (0.0, 1.0), p["frequency_mod_depth"])
                mod = Sine(freq=rate, mul=depth * oscillators["positive_emotion"].freq)
                modulators[group] = mod
                oscillators["positive_emotion"].freq = oscillators["positive_emotion"].freq + mod

            if group == "provocation":
                amp = scale_value(val, (0.0, 1.0), p["amplitude"])
                dist = scale_value(val, (0.0, 1.0), p["distortion"])
                for nm, osci in oscillators.items():
                    if dist > 0:
                        oscillators[nm] = Disto(osci, drive=dist, slope=0.8, mul=amp)
                    else:
                        osci.mul = amp

            if group == "style":
                tempo = scale_value(val, (0.0, 1.0), p["tempo"])
                trem = Sine(freq=tempo, mul=0.3, add=0.7)
                for osci in oscillators.values():
                    osci.mul = osci.mul * trem

            if group == "topic":
                filt_freq = scale_value(val, (0.0, 1.0), p["filter_freq"])
                rev_amt = scale_value(val, (0.0, 1.0), p["reverb"])
                mix = Mix(list(oscillators.values()), voices=1)
                filt = Biquadx(mix, freq=filt_freq, q=1.0, type=0)
                if rev_amt > 0:
                    filt = WGVerb(filt, feedback=rev_amt, cutoff=5000.0, bal=0.3)
                filters[group] = filt

        final = filters[list(filters.keys())[-1]] if filters else Mix(list(oscillators.values()), voices=1)
        for osci in oscillators.values(): osci.play()
        for mod in modulators.values(): mod.play()
        final.out()

        table = NewTable(length=duration_per_row, chnls=1)
        TableRec(final, table).play()
        time.sleep(duration_per_row)
        all_segments.append(np.copy(table.getTable()))

        for osci in oscillators.values(): osci.stop()
        for mod in modulators.values(): mod.stop()
        for filt in filters.values(): filt.stop()
        final.stop()

    server.stop()
    server.shutdown()

    if not all_segments:
        return np.zeros(int(sr * duration_per_row))
    waveform = np.concatenate(all_segments)
    max_amp = np.max(np.abs(waveform))
    if max_amp > 0:
        waveform = waveform / max_amp * 0.9
    return waveform


def sonify_example(save_path=None):
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    size = 20
    df = pd.DataFrame({
        'emotion_hope': np.random.rand(size), 'emotion_pride': np.random.rand(size),
        'emotion_anger': np.random.rand(size), 'emotion_fear': np.random.rand(size), 'emotion_despair': np.random.rand(size),
        'sentiment_polarity': np.random.uniform(-1,1,size), 'moral_outrage': np.random.rand(size),
        'intent_provocation': np.random.rand(size), 'hostility': np.random.rand(size),
        'style_urgency': np.random.rand(size), 'style_informality': np.random.rand(size), 'assertion_strength': np.random.rand(size),
        'topic_security': np.random.rand(size), 'topic_resource': np.random.rand(size), 'military_intensity': np.random.rand(size)
    })
    waveform = create_sonification(df, duration_per_row=0.3)
    if save_path:
        try:
            import soundfile as sf
            sf.write(save_path, waveform, 44100)
        except ImportError:
            print("Install soundfile to save: pip install soundfile")
    return waveform


def sonify_example_2(save_path=None, duration_per_row=0.3):
    """
    Generate and optionally save an example sonification.
    
    Args:
        save_path: Optional path to save the generated audio (e.g., "example.wav")
        duration_per_row: Duration in seconds for each data row
        
    Returns:
        np.ndarray: The generated audio waveform
        
    """
    import pandas as pd
    import numpy as np
    
    # Generate example data
    np.random.seed(42)
    size = 20
    columns = [
        'emotion_hope', 'emotion_pride',
        'emotion_anger', 'emotion_fear', 'emotion_despair',
        'sentiment_polarity', 'moral_outrage',
        'intent_provocation', 'hostility',
        'style_urgency', 'style_informality', 'assertion_strength',
        'topic_security', 'topic_resource', 'military_intensity'
    ]
    
    # Create a DataFrame with random values
    data = {col: np.random.rand(size) for col in columns}
    # Special case for sentiment polarity which ranges from -1 to 1
    data['sentiment_polarity'] = np.random.uniform(-1, 1, size)
    
    df = pd.DataFrame(data)
    
    # Generate the waveform

    waveform = create_sonification(df, duration_per_row=duration_per_row)
    
    # Save if requested
    if save_path:
        try:
            import soundfile as sf
            sf.write(save_path, waveform, 44100)
            print(f"Saved sonification to {save_path}")
        except ImportError:
            print("Could not save audio: Please install soundfile with 'pip install soundfile'")
        except Exception as e:
            print(f"Error saving audio to {save_path}: {e}")
    
    return waveform
