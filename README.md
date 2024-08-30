
# sonification
 Map data to sound allowing it to be interpreted it in an auditory manner


To install:	```pip install sonification```

# Examples

```python
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
        df['duration'] = df['duration'] + 0.01  # Adding a small value to ensure durations are not zero
    
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
        
        # print(f"Row {index}: pitch={pitch}, duration={duration}, volume={volume}, frequency={frequency}, wave_len={len(wave)}")
        
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

def sonification_dataframe(df, pitch_col, duration_col, volume_col, sample_rate=44100, filepath=None):
    df, label_encoders = preprocess_dataframe(df)
    waveform, sr = map_features_to_audio(df, pitch_col, duration_col, volume_col, sample_rate)
    return save_or_return_audio(waveform, sr, filepath)

# Example usage:
df = pd.DataFrame({
    'pitch': [0.2, 0.4, 0.6, 0.8],
    'duration': [0.5, 0.5, 0.5, 0.5],
    'volume': [0.5, 0.7, 0.9, 1.0]
})
waveform, sr = sonification_dataframe(df, 'pitch', 'duration', 'volume')

# To play the audio
if waveform is not None and len(waveform) > 0:
    play_obj = sa.play_buffer(np.int16(waveform / np.max(np.abs(waveform)) * 32767), 1, 2, sr)
    play_obj.wait_done()
```