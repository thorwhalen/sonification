import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import simpleaudio as sa


def preprocess_dataframe(df):
    df = df.copy()
    # Normalize numerical columns
    scaler = MinMaxScaler()
    for column in df.select_dtypes(include=np.number).columns:
        df[column] = scaler.fit_transform(df[[column]])

    # Ensure durations are not zero
    if 'duration' in df.columns:
        df['duration'] = (
            df['duration'] + 0.01
        )  # Adding a small value to ensure durations are not zero

    # Encode categorical columns
    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    return df, label_encoders


def generate_tone(frequency, duration, volume, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = volume * np.sin(frequency * t * 2 * np.pi)
    return wave


def map_features_to_audio(df, pitch_col, duration_col, volume_col, sample_rate=44100):
    waveform = np.array([])

    for index, row in df.iterrows():
        pitch = row[pitch_col]
        duration = row[duration_col]
        volume = row[volume_col]

        frequency = 440 + pitch * 440  # Example: Map pitch to frequency
        wave = generate_tone(frequency, duration, volume, sample_rate)

        print(
            f"Row {index}: pitch={pitch}, duration={duration}, volume={volume}, frequency={frequency}, wave_len={len(wave)}"
        )

        waveform = np.concatenate([waveform, wave])

    print(f"Final waveform length: {len(waveform)}")
    return waveform, sample_rate


def save_or_return_audio(waveform, sample_rate, filepath=None):
    if filepath:
        # Normalize waveform to int16 range
        waveform_int16 = np.int16(waveform / np.max(np.abs(waveform)) * 32767)
        sa.WaveObject(waveform_int16, 1, 2, sample_rate).save(filepath)
    else:
        return waveform, sample_rate


def sonification_dataframe(
    df, pitch_col, duration_col, volume_col, sample_rate=44100, filepath=None
):
    df, label_encoders = preprocess_dataframe(df)
    waveform, sr = map_features_to_audio(
        df, pitch_col, duration_col, volume_col, sample_rate
    )
    return save_or_return_audio(waveform, sr, filepath)


# -----------------------------------------------------
from contextlib import suppress


with suppress(ImportError, ModuleNotFoundError):
    import numpy as np
    import pandas as pd
    from astropy.table import Table
    from typing import Optional, Dict, Union, Tuple, List
    from collections.abc import Callable
    from astronify.series import SoniSeries
    from scipy import signal
    import scipy.io.wavfile
    import warnings


    import warnings

    # Filter warnings
    warnings.filterwarnings("ignore", message="Unknown midi type")


    def sonify_dataframe_w_astronify(
        df: pd.DataFrame,
        time_col: str | None = None,
        pitch_cols: str | list[str] | None = None,
        egress: str | None = None,
        note_spacing: float = 0.01,
        note_duration: float = 0.5,
        gain: float = 0.05,
        pitch_range: tuple[float, float] = (100, 10000),
        center_pitch: float = 440,
        zero_point: str | float = "median",
        stretch: str = "linear",
        minmax_percent: list[float] | None = None,
        minmax_value: list[float] | None = None,
        invert: bool = False,
        custom_mapper: Callable | None = None,
        combine: bool = True,
        weights: dict[str, float] | None = None,
        sample_rate: int = 44100,
        waveform_type: str = "sine"
    ) -> np.ndarray | dict[str, dict]:
        """
        Sonify a pandas DataFrame using astronify for pitch mapping and scipy for waveform generation.
        
        This function converts a pandas DataFrame into audio, with each numeric column
        sonified separately and optionally combined into a single waveform.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to sonify.
        time_col : str, optional
            Column to use as time values. If None, an index from 0 to len(df) will be created.
        pitch_cols : str or list of str, optional
            Column(s) to map to pitch. If None, all columns except time_col will be used.
        egress : str, optional
            Base filename to save the audio. Will append column name for each pitch column
            if combine=False, or save as a single file if combine=True.
            If None, no file is saved and the waveform is returned.
        note_spacing : float, default 0.01
            Spacing between notes in seconds.
        note_duration : float, default 0.5
            Duration of each note in seconds.
        gain : float, default 0.05
            Audio gain (0.0 to 1.0).
        pitch_range : tuple of float, default (100, 10000)
            (min, max) range of pitch values in Hz.
        center_pitch : float, default 440
            The reference pitch in Hz where the zero_point will be mapped.
        zero_point : str or float, default "median"
            Data value mapped to center_pitch. Options: "mean", "median", or a float.
        stretch : str, default "linear"
            Stretch to apply to values. Options: "linear", "log", "sqrt", "asinh", "sinh".
        minmax_percent : list of float, optional
            [min_percentile, max_percentile] to clip data values.
        minmax_value : list of float, optional
            [min_value, max_value] to clip data values.
        invert : bool, default False
            If True, invert pitch mapping (higher values = lower pitch).
        custom_mapper : callable, optional
            A custom function to map data values to pitch, replacing the default.
        combine : bool, default True
            If True, combines all sonifications into a single waveform.
            If False, returns individual sonifications.
        weights : Dict[str, float], optional
            Dictionary mapping column names to their relative weights when combining.
            Only used if combine=True.
        sample_rate : int, default 44100
            Sample rate for the audio in Hz.
        waveform_type : str, default "sine"
            Type of waveform to generate. Options: "sine", "square", "sawtooth", "triangle".
            
        Returns
        -------
        Union[numpy.ndarray, Dict[str, Dict]]
            If combine=True, returns the combined waveform as a numpy array.
            If combine=False, returns a dictionary mapping column names to their sonification data.
            Each sonification contains 'waveform', 'pitch_values', and 'onsets' keys.
        """
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Create time column if not specified
        if time_col is None:
            time_values = np.arange(len(df))
            time_series = pd.Series(time_values, index=df.index, name='time')
            df = df.copy()
            df['time'] = time_series
            time_col = 'time'
        elif time_col == 'index':
            # Special handling for the DataFrame index
            df = df.reset_index().copy()
            time_col = 'index'
        elif time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame")
        
        # Determine pitch columns
        if pitch_cols is None:
            pitch_cols = [col for col in df.columns if col != time_col]
        elif isinstance(pitch_cols, str):
            pitch_cols = [pitch_cols]
        
        for col in pitch_cols:
            if col not in df.columns:
                raise ValueError(f"Pitch column '{col}' not found in DataFrame")
        
        # Ensure data is numeric
        for col in [time_col] + pitch_cols:
            if not np.issubdtype(df[col].dtype, np.number):
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    raise ValueError(f"Column '{col}' could not be converted to numeric")
        
        # Create pitch mapping parameters
        pitch_map_args = {
            "pitch_range": pitch_range,
            "center_pitch": center_pitch,
            "zero_point": zero_point,
            "stretch": stretch,
            "invert": invert
        }
        
        if minmax_percent is not None:
            pitch_map_args["minmax_percent"] = minmax_percent
        
        if minmax_value is not None:
            pitch_map_args["minmax_value"] = minmax_value
        
        # Create sonifications for each pitch column
        sonifications = {}
        print(f"Processing {len(pitch_cols)} columns...")
        
        for col in pitch_cols:
            # print(f"Sonifying column: {col}")
            try:
                # Create Astropy table with time and current column
                table_data = Table({
                    time_col: df[time_col].values,
                    'flux': df[col].values
                })
                
                # Create SoniSeries object to use astronify's pitch mapping only
                soni_obj = SoniSeries(table_data, time_col=time_col, val_col='flux')
                
                # Set sonification parameters
                soni_obj.note_spacing = note_spacing
                soni_obj.note_duration = note_duration
                soni_obj.gain = gain
                
                # Apply custom pitch mapper if provided
                if custom_mapper is not None:
                    soni_obj.pitch_mapper.pitch_map_func = custom_mapper
                
                # Update pitch mapping arguments
                soni_obj.pitch_mapper.pitch_map_args.update(pitch_map_args)
                
                # Perform sonification to get pitch values only (no audio generation yet)
                soni_obj.sonify()
                
                # Get the pitch values and onset times
                pitch_values = soni_obj.data['asf_pitch'].data  # Frequencies in Hz
                onsets = soni_obj.data['asf_onsets'].data      # Onset times in seconds
                
                # print(f"  - Generated {len(pitch_values)} pitch values")
                
                # Generate the waveform using scipy (NOT using Pyo)
                waveform = generate_waveform(
                    pitch_values, 
                    onsets, 
                    note_duration, 
                    sample_rate, 
                    gain,
                    waveform_type
                )
                
                # print(f"  - Created waveform of length {len(waveform)}")
                
                # Store in results
                sonifications[col] = {
                    'waveform': waveform,
                    'pitch_values': pitch_values,
                    'onsets': onsets
                }
                
                # Save to file if requested and not combining
                if egress is not None and not combine:
                    filename = f"{egress}_{col}.wav"
                    # Convert to int16 for WAV file compatibility
                    waveform_int = (waveform * 32767).astype(np.int16)
                    scipy.io.wavfile.write(filename, sample_rate, waveform_int)
                    print(f"  - Saved to {filename}")
                    
            except Exception as e:
                print(f"Error processing column '{col}': {str(e)}")
                continue
        
        # Check if we have any successful sonifications
        if not sonifications:
            raise RuntimeError("Failed to generate any sonifications")
        
        # Return individual sonifications or combined waveform
        if not combine:
            print(f"Returning {len(sonifications)} individual sonifications")
            return sonifications
        else:
            # Combine the sonifications
            print(f"Combining {len(sonifications)} sonifications...")
            waveform = combine_waveforms(sonifications, weights)
            
            # Save to file if requested
            if egress is not None:
                # Convert to int16 for WAV file compatibility
                waveform_int = (waveform * 32767).astype(np.int16)
                scipy.io.wavfile.write(egress, sample_rate, waveform_int)
                print(f"Saved combined audio to {egress}")
            
            print(f"Returning combined waveform of length {len(waveform)}")
            return waveform


    def generate_waveform(
        frequencies: np.ndarray, 
        onsets: np.ndarray, 
        note_duration: float,
        sample_rate: int = 44100,
        gain: float = 0.05,
        waveform_type: str = "sine"
    ) -> np.ndarray:
        """
        Generate a waveform from frequencies and onset times.
        
        Parameters
        ----------
        frequencies : numpy.ndarray
            Array of frequencies in Hz.
        onsets : numpy.ndarray
            Array of onset times in seconds.
        note_duration : float
            Duration of each note in seconds.
        sample_rate : int, default 44100
            Sample rate in Hz.
        gain : float, default 0.05
            Amplitude of the waveform.
        waveform_type : str, default "sine"
            Type of waveform to generate.
            
        Returns
        -------
        numpy.ndarray
            The generated waveform.
        """
        if len(frequencies) != len(onsets):
            raise ValueError("Frequencies and onsets must have the same length")
        
        if len(frequencies) == 0:
            return np.array([])
        
        # Calculate the total duration of the waveform
        total_duration = onsets[-1] + note_duration if len(onsets) > 0 else 0
        
        # Create the output array
        num_samples = int(total_duration * sample_rate)
        output = np.zeros(num_samples, dtype=np.float32)
        
        # Generate the waveform for each note
        for freq, onset in zip(frequencies, onsets):
            # Convert onset time to sample index
            onset_sample = int(onset * sample_rate)
            
            # Calculate end of note
            end_sample = min(onset_sample + int(note_duration * sample_rate), num_samples)
            
            # Calculate duration in samples
            dur_samples = end_sample - onset_sample
            
            if dur_samples <= 0:
                continue
            
            # Generate the waveform
            t = np.linspace(0, note_duration, dur_samples)
            
            # Create envelope for the note (ADSR: Attack, Decay, Sustain, Release)
            attack_dur = 0.1  # 10% of note duration for attack
            decay_dur = 0.1   # 10% of note duration for decay
            release_dur = 0.2  # 20% of note duration for release
            
            attack_samples = int(attack_dur * dur_samples)
            decay_samples = int(decay_dur * dur_samples)
            release_samples = int(release_dur * dur_samples)
            sustain_samples = dur_samples - attack_samples - decay_samples - release_samples
            
            env = np.ones(dur_samples)
            
            # Attack phase
            if attack_samples > 0:
                env[:attack_samples] = np.linspace(0, 1, attack_samples)
            
            # Decay phase
            if decay_samples > 0:
                decay_end = attack_samples + decay_samples
                env[attack_samples:decay_end] = np.linspace(1, 0.8, decay_samples)
            
            # Release phase
            if release_samples > 0:
                env[-release_samples:] = np.linspace(0.8, 0, release_samples)
            
            # Generate the appropriate waveform
            if waveform_type == "sine":
                wave = np.sin(2 * np.pi * freq * t)
            elif waveform_type == "square":
                wave = signal.square(2 * np.pi * freq * t)
            elif waveform_type == "sawtooth":
                wave = signal.sawtooth(2 * np.pi * freq * t)
            elif waveform_type == "triangle":
                wave = signal.sawtooth(2 * np.pi * freq * t, width=0.5)
            else:
                wave = np.sin(2 * np.pi * freq * t)  # Default to sine
            
            # Apply envelope to the waveform
            wave = wave * env * gain
            
            # Add to output
            output[onset_sample:end_sample] += wave
        
        return output


    def combine_waveforms(
        sonifications: dict[str, dict],
        weights: dict[str, float] | None = None
    ) -> np.ndarray:
        """
        Combine multiple sonifications into a single waveform.
        
        Parameters
        ----------
        sonifications : Dict[str, Dict]
            Dictionary of sonifications as returned by sonify_dataframe_w_astronify with combine=False.
        weights : Dict[str, float], optional
            Dictionary mapping column names to their relative weights in the mix.
            
        Returns
        -------
        numpy.ndarray
            The combined waveform as a numpy array.
        """
        if not sonifications:
            raise ValueError("No sonifications provided")
        
        # Default weights if not specified
        if weights is None:
            weights = {col: 1.0 for col in sonifications}
        else:
            # Check that weights are provided for all sonifications
            for col in sonifications:
                if col not in weights:
                    weights[col] = 1.0
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Weights sum to zero")
        normalized_weights = {col: w / total_weight for col, w in weights.items()}
        
        # Find the maximum length
        max_length = max(len(soni['waveform']) for soni in sonifications.values())
        
        # Initialize the output array
        output = np.zeros(max_length, dtype=np.float32)
        
        # Add each sonification with its weight
        for col, soni in sonifications.items():
            waveform = soni['waveform']
            weight = normalized_weights[col]
            
            # Pad the waveform if necessary
            if len(waveform) < max_length:
                padded = np.zeros(max_length, dtype=np.float32)
                padded[:len(waveform)] = waveform
                waveform = padded
            
            # Add to output
            output += waveform * weight
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(output))
        if max_amplitude > 1.0:
            output = output / max_amplitude
        
        return output
