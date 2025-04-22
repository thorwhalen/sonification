#!/usr/bin/python
import os

from pyo import *
from sonification.pyo_util import Server


def sonify_parameters(
    parameter_frames, frame_durations=1.0, output_filename="sonification_output.wav"
):
    if isinstance(frame_durations, (float, int)):
        durations = [frame_durations] * len(parameter_frames)
    elif len(frame_durations) != len(parameter_frames):
        raise ValueError(
            "frame_durations must be a float or a list with the same length as parameter_frames."
        )
    else:
        durations = frame_durations

    total_duration = sum(durations)

    # Boot offline server
    s = Server(audio="offline").boot()
    table_size = int(s.getSamplingRate() * total_duration)
    tbl = NewTable(length=total_duration)

    # Signal objects
    freq = SigTo(value=440, time=0.05)
    amp = SigTo(value=0.5, time=0.05)
    cutoff = SigTo(value=1000, time=0.05)
    tempo = SigTo(value=120, time=0.05)

    # Base oscillator
    osc = Sine(freq=freq)

    # Pulse modulator synced to tempo (tempo/60 Hz = beats per second)
    pulse_rate = tempo / 60.0
    lfo = Sine(freq=pulse_rate).range(
        0.2, 1.0
    )  # Can tweak range for stronger/weaker effect

    # Multiply audio by amplitude and tempo-pulse
    modulated = osc * amp * lfo
    filtered = ButLP(modulated, freq=cutoff)

    # Record to table
    rec = TableRec(filtered, table=tbl).play()

    # Step through each frame
    for frame, dur in zip(parameter_frames, durations):
        freq.value = frame['frequency']
        amp.value = frame['amplitude']
        cutoff.value = frame['cutoff']
        tempo.value = frame['tempo']
        s.recordOptions(dur=dur)
        s.start()

    rec.stop()
    tbl.save(output_filename)

    print(
        f"✅ Audio rendered with tempo pulses and saved to {os.path.abspath(output_filename)}"
    )


if __name__ == "__main__":
    import sys

    output_filename = sys.argv[1]

    parameter_frames = [
        {'tempo': 120, 'frequency': 440, 'amplitude': 0.5, 'cutoff': 1000},
        {'tempo': 135, 'frequency': 330, 'amplitude': 0.7, 'cutoff': 500},
        {'tempo': 100, 'frequency': 550, 'amplitude': 0.3, 'cutoff': 2000},
        {'tempo': 150, 'frequency': 220, 'amplitude': 0.9, 'cutoff': 750},
        {'tempo': 110, 'frequency': 660, 'amplitude': 0.6, 'cutoff': 1500},
    ]
    sonify_parameters(
        parameter_frames, frame_durations=2.0, output_filename=output_filename
    )
