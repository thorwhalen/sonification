"""Dynamic pyo example"""

from pyo import *


def make_synth_voice(pitch, volume, waveform="sine"):
    """
    Returns a dynamic audio voice controlled by `pitch` and `volume`.

    Parameters:
        pitch (PyoObject or float): Frequency in Hz.
        volume (PyoObject or float): Amplitude scalar.
        waveform (str): "sine", "square", or "saw".

    Returns:
        PyoObject: Output signal (mono).
    """
    freq = Sig(pitch)
    amp = Sig(volume)

    if waveform == "sine":
        osc = Sine(freq=freq, mul=amp)
    elif waveform == "square":
        osc = LFO(freq=freq, type=1, mul=amp)
    elif waveform == "saw":
        osc = LFO(freq=freq, type=2, mul=amp)
    else:
        raise ValueError(f"Unknown waveform: {waveform}")

    env = Fader(fadein=0.01, fadeout=0.1, dur=4).play()
    return osc * env


# === OFFLINE RENDERING ===
# `duplex=0` disables input, and `audio="offline"` enables file rendering
s = Server(duplex=0, audio="offline").boot()

# Duration of the render
duration = 4

# Output file setup
s.recordOptions(dur=duration, filename="test.wav", fileformat=0, sampletype=0)

# Control inputs
pitch = SigTo(value=440, time=0.1)
volume = SigTo(value=0.3, time=0.1)

# Synth output
voice = make_synth_voice(pitch, volume, waveform="saw").out()

# Start rendering — no need to call `.record()` here
s.start()
s.shutdown()

print("✅ Render complete! Audio saved to 'test.wav'")
