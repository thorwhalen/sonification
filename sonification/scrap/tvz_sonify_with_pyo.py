"""
Data Sonification with Pyo

This module converts multidimensional data into sound parameters, creating
a sonification of the data attributes.

Example usage:
    # Assuming score_df already loaded
    waveform = create_sonification(score_df)
    import soundfile as sf
    sf.write("sonification.wav", waveform, 44100)
"""

from typing import Dict, List, Tuple, Mapping, Iterable, Sequence, Iterator
import time

import numpy as np
from pyo import *


def scale_value(
    value: float, source_range: Tuple[float, float], target_range: Tuple[float, float]
) -> float:
    """
    Scale a value from source range to target range.

    >>> scale_value(0.5, (0, 1), (100, 500))
    300.0
    """
    src_min, src_max = source_range
    tgt_min, tgt_max = target_range

    # Handle the case where source range is a single point
    if src_max == src_min:
        normalized = 0.5
    else:
        normalized = (value - src_min) / (src_max - src_min)

    return tgt_min + normalized * (tgt_max - tgt_min)


def _compute_group_value(row: Mapping, attributes: Sequence[str]) -> float:
    """
    Compute the combined value for a group of attributes.

    Takes the mean of positive attributes and subtracts the mean of negative attributes.
    """
    # In a more complex implementation, you could weight attributes differently
    # or use a more sophisticated combination method
    return np.mean([row[attr] for attr in attributes])


class Osc303(PyoObject):
    """Basic emulation of a TB-303"""

    def __init__(self, freq=20, decay=0.5, shape=1, cutoff=2000, reso=5, mul=1, add=0):
        # Initialize the base class
        PyoObject.__init__(self, mul, add)

        # Store parameters
        self._freq = freq
        self._shape = shape
        self._cutoff = cutoff
        self._reso = reso
        self._decay = decay

        # For backward compatibility
        # self.freq = freq
        self.mul = mul

        # Create DSP chain
        self.wave1 = Phasor(freq=self._freq, phase=0, mul=1)
        self.wave2 = Phasor(freq=self._freq * (-1), phase=0.5, mul=1)
        self.env = Adsr(
            0.05, self._decay, sustain=0.1, release=0.05, dur=0.2 + self._decay
        )
        self.env2 = self.env * self.mul

        shape_0_wave3 = Phasor(freq=self._freq, phase=0, mul=self.env2)
        shape_1_wave3 = ((self.wave1 + self.wave2) - 1) * self.env2
        # Linear interpolation between the two waves based on shape
        self.wave3 = shape_0_wave3 * (1 - self._shape) + shape_1_wave3 * self._shape

        self.filter = Biquadx(
            [self.wave3, self.wave3], freq=self._cutoff, q=self._reso, type=0, stages=2
        )

        # Register the output signal for PyoObject's processing
        self._base_objs = self.filter.getBaseObjects()

    def play(self, *args, **kwargs):
        self.env.play(*args, **kwargs)
        return PyoObject.play(self, *args, **kwargs)

    def stop(self):
        self.filter.stop()
        return PyoObject.stop(self)

    def out(self, *args, **kwargs):
        self.filter.out(*args, **kwargs)
        return PyoObject.out(self, *args, **kwargs)

    def setFreq(self, x):
        """Replace the `freq` attribute."""
        self._freq = x
        self.wave1.freq = x
        self.wave2.freq = x * (-1)

    @property
    def freq(self):
        """Get the frequency of the oscillator."""
        return self._freq

    @freq.setter
    def freq(self, x):
        """Set the frequency of the oscillator."""
        self.setFreq(x)  # Use the existing method


def create_sonification(data_df, duration_per_row=0.5, sr=44100) -> np.ndarray:
    """
    Create a sonification of the data.

    Args:
        data_df: DataFrame with attribute columns
        duration_per_row: Duration in seconds for each data row
        sr: Sample rate

    Returns:
        A numpy array containing the audio waveform samples
    """
    # Define attribute groups and their corresponding audio parameters
    attribute_groups = {
        "positive_emotion": ["emotion_hope", "emotion_pride"],
        "negative_emotion": ["emotion_anger", "emotion_fear", "emotion_despair"],
        "sentiment": ["sentiment_polarity", "moral_outrage"],
        "provocation": ["intent_provocation", "hostility"],
        "style": ["style_urgency", "style_informality", "assertion_strength"],
        "topic": ["topic_security", "topic_resource", "military_intensity"],
    }

    # Map each attribute group to sonic parameters
    # Each parameter gets a (min, max) range for mapping
    param_mappings = {
        "positive_emotion": {
            "frequency": (300, 800),  # Frequency in Hz (mid-to-high range)
            "harmonic_content": (0.1, 0.4),  # Consonant sounds
            "attack": (0.1, 0.3),  # Gentle attack
            "decay": (0.5, 1.5),  # Moderate decay
        },
        "negative_emotion": {
            "frequency": (100, 500),  # Lower frequency range
            "harmonic_content": (0.4, 0.9),  # More dissonant
            "attack": (0.01, 0.1),  # Sharper attack
            "decay": (0.3, 0.8),  # Shorter decay
        },
        "sentiment": {
            "frequency_mod": (0, 8),  # Frequency modulation rate
            "frequency_mod_depth": (0, 0.1),  # Depth of frequency modulation
        },
        "provocation": {
            "amplitude": (0.3, 0.8),  # Overall volume
            "distortion": (0, 0.7),  # Amount of distortion
        },
        "style": {
            "tempo": (1, 8),  # Notes per second
            "resonance": (0.1, 5),  # Filter resonance
        },
        "topic": {
            "filter_freq": (500, 5000),  # Filter cutoff frequency
            "reverb": (0, 0.5),  # Amount of reverb
        },
    }

    # Initialize pyo server in offline mode
    server = Server(audio="offline", sr=sr, nchnls=1, buffersize=512, duplex=0).boot()

    # Create a list to hold all the recorded segments
    all_segments = []

    # Process each row of the dataframe
    for i in range(len(data_df)):
        row = data_df.iloc[i].to_dict()

        # Create sound generators for each attribute group
        oscillators = {}
        modulators = {}
        filters = {}

        # Set up the audio chain for each attribute group
        for group_name, attributes in attribute_groups.items():
            # Calculate the combined value for this group
            group_value = _compute_group_value(row, attributes)

            # Get the parameter mapping for this group
            params = param_mappings[group_name]

            # Create audio generators based on the group
            if group_name in ["positive_emotion", "negative_emotion"]:
                # Base frequency determined by the group value
                freq = float(scale_value(group_value, (0, 1), params["frequency"]))

                # Harmonic content affects the waveform shape
                shape = float(
                    scale_value(group_value, (0, 1), params["harmonic_content"])
                )

                # ADSR envelope parameters - convert NumPy values to Python floats
                attack = float(scale_value(group_value, (0, 1), params["attack"]))
                decay = float(scale_value(group_value, (0, 1), params["decay"]))

                # Create oscillator with envelope
                env = Adsr(
                    attack=attack,
                    decay=decay,
                    sustain=0.3,
                    release=0.5,
                    dur=duration_per_row,
                    mul=0.5,
                )

                # Use the Osc303 oscillator which has good control over harmonic content
                oscillators[group_name] = Osc303(
                    freq=freq, decay=decay, shape=shape, cutoff=3000, reso=1, mul=env
                )

            if group_name == "sentiment":
                # Add frequency modulation to existing oscillators
                if "positive_emotion" in oscillators:
                    # Use sentiment to create frequency modulation
                    mod_rate = float(
                        scale_value(group_value, (0, 1), params["frequency_mod"])
                    )
                    mod_depth = float(
                        scale_value(group_value, (0, 1), params["frequency_mod_depth"])
                    )

                    # Create LFO for frequency modulation
                    modulators[group_name] = Sine(
                        freq=mod_rate,
                        mul=mod_depth * oscillators["positive_emotion"].freq,
                    )

                    # Apply the modulation
                    oscillators["positive_emotion"].freq = (
                        oscillators["positive_emotion"].freq + modulators[group_name]
                    )

            if group_name == "provocation":
                # Apply amplitude and distortion to all oscillators
                amp = float(scale_value(group_value, (0, 1), params["amplitude"]))
                dist = float(scale_value(group_value, (0, 1), params["distortion"]))

                for osc_name in oscillators:
                    # Apply distortion - simple tanh-based distortion
                    if dist > 0:
                        oscillators[osc_name] = Disto(
                            oscillators[osc_name], drive=dist, slope=0.8, mul=amp
                        )
                    else:
                        # Just adjust amplitude
                        oscillators[osc_name].mul = amp

            if group_name == "style":
                # Apply style parameters across all sounds
                resonance = float(scale_value(group_value, (0, 1), params["resonance"]))

                # Create a pulse with tempo based on style
                tempo = float(scale_value(group_value, (0, 1), params["tempo"]))

                # Use the tempo to affect the tremolo of the sound
                trem = Sine(freq=tempo, mul=0.3, add=0.7)

                # Apply to all oscillators
                for osc_name in oscillators:
                    # Add tremolo effect
                    oscillators[osc_name].mul = oscillators[osc_name].mul * trem

            if group_name == "topic":
                # Filter frequency based on topic
                filter_freq = float(
                    scale_value(group_value, (0, 1), params["filter_freq"])
                )
                reverb_amt = float(scale_value(group_value, (0, 1), params["reverb"]))

                # Create a mixer to combine all oscillators
                mix = Mix([osc for osc in oscillators.values()], voices=1)

                # Apply a filter
                filters[group_name] = Biquad(
                    mix, freq=filter_freq, q=1, type=0  # 0=lowpass
                )

                # Add reverb if needed
                if reverb_amt > 0:
                    filters[group_name] = WGVerb(
                        filters[group_name], feedback=reverb_amt, cutoff=5000, bal=0.3
                    )

        # If we created filters in the topic stage, use that as the final output
        # Otherwise mix all oscillators
        final_output = None
        if filters:
            # Use the last filter created as output
            final_output = filters[list(filters.keys())[-1]]
            final_output.out()
        else:
            # Mix all oscillators
            final_output = Mix([osc for osc in oscillators.values()], voices=1)
            final_output.out()

        # Configure server to record for the specified duration
        server.recordOptions(dur=duration_per_row)

        # Start the server and record
        server.start()

        # Wait for the recording to complete
        # time.sleep(duration_per_row)

        # Get the recorded data from the table
        total_duration = duration_per_row * len(data_df)
        table = NewTable(length=total_duration, chnls=1)
        rec = TableRec(final_output, table).play()

        # table = NewTable(length=duration_per_row, chnls=1)
        # rec = TableRec(final_output, table).play()
        time.sleep(duration_per_row)
        samples = table.getTable()
        all_segments.append(samples)

        # Stop server for next iteration
        server.stop()

        # Reset all audio objects to avoid memory issues
        for osc in oscillators.values():
            osc.stop()
        for mod in modulators.values():
            mod.stop()
        for filt in filters.values():
            filt.stop()
        if final_output:
            final_output.stop()

    # Shutdown server
    server.shutdown()

    # Combine all segments into one waveform
    if not all_segments:
        return np.zeros(100)  # Return empty array if no segments were created

    final_waveform = np.concatenate(all_segments)

    # Normalize the waveform
    if np.max(np.abs(final_waveform)) > 0:
        final_waveform = final_waveform / np.max(np.abs(final_waveform)) * 0.9

    return final_waveform


# Example usage with multiple rows
def sonify_example(save_path=None):
    """
    Example of sonifying a dataframe with multiple rows.

    Args:
        save_path: Optional path to save the waveform as a WAV file

    Returns:
        The generated waveform as a numpy array
    """
    import pandas as pd

    # Create sample data (this would be your score_df)
    np.random.seed(42)
    sample_size = 20
    sample_df = pd.DataFrame(
        {
            # Emotion attributes
            'emotion_hope': np.random.uniform(0, 1, sample_size),
            'emotion_pride': np.random.uniform(0, 1, sample_size),
            'emotion_anger': np.random.uniform(0, 1, sample_size),
            'emotion_fear': np.random.uniform(0, 1, sample_size),
            'emotion_despair': np.random.uniform(0, 1, sample_size),
            # Sentiment attributes
            'sentiment_polarity': np.random.uniform(-1, 1, sample_size),
            'moral_outrage': np.random.uniform(0, 1, sample_size),
            # Provocation attributes
            'intent_provocation': np.random.uniform(0, 1, sample_size),
            'hostility': np.random.uniform(0, 1, sample_size),
            # Style attributes
            'style_urgency': np.random.uniform(0, 1, sample_size),
            'style_informality': np.random.uniform(0, 1, sample_size),
            'assertion_strength': np.random.uniform(0, 1, sample_size),
            # Topic attributes
            'topic_security': np.random.uniform(0, 1, sample_size),
            'topic_resource': np.random.uniform(0, 1, sample_size),
            'military_intensity': np.random.uniform(0, 1, sample_size),
        }
    )

    # Create sonification
    waveform = create_sonification(sample_df, duration_per_row=0.3)

    # Save to file if a path is provided
    if save_path:
        try:
            import soundfile as sf

            sf.write(save_path, waveform, 44100)
            print(f"Saved sonification to {save_path}")
        except ImportError:
            print(
                "Soundfile library not available. Install with: pip install soundfile"
            )
            print(f"Waveform generated with shape {waveform.shape}")

    return waveform


if __name__ == "__main__":
    sonify_example("sonification_example.wav")
