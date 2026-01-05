"""
Emotion Music Generator

A customizable music generation tool that creates MIDI music based on emotion parameters.
This module uses music21 to generate melodies and accompaniments that reflect emotional
timeseries data, with enhanced melodic and harmonic variation.

Key Components:
- Feature Mapping: Maps raw emotion features to musical dimensions (valence, arousal, etc.)
- Scale Selection: Customizable scales for different emotional states
- Chord Progression: Dynamic and varied chord progressions based on emotional states
- Melodic Generation: Creates varied melodies reflecting emotional parameters using
  generative algorithms rather than fixed patterns
- Musical Parameters: Customizable tempo, note durations, and dynamics

Main Parameters:
- scale: Base scale to use for composition (default C Major)
- phrase_duration: Duration of each emotion segment in beats
- tempo_range: Range of tempos to map arousal to
- feature_mapping: How to group emotion features into musical dimensions
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import (
    List,
    Dict,
    Optional,
    Union,
    Tuple,
    Any,
    Set,
)
from collections.abc import Callable, Generator, Iterator
import time
import random

# Default module constants - can be modified if needed
DEFAULT_SCALE = [0, 2, 4, 5, 7, 9, 11]  # C Major (0=C, 1=C#, etc.)
DEFAULT_SCALE_ROOT = 0  # C
DEFAULT_TEMPO_RANGE = (60, 160)  # BPM range mapped to arousal
DEFAULT_PHRASE_DURATION = 4.0  # Beats per emotion data point
DEFAULT_OCTAVE_RANGE = (3, 5)  # Base octave range for melody
DEFAULT_VELOCITY_RANGE = (70, 110)  # MIDI velocity range for dynamics
DEFAULT_DURATION_MAPPINGS = {
    'high_arousal': [0.25, 0.25, 0.5, 0.25, 0.25, 0.5],  # Fast notes
    'medium_arousal': [0.5, 0.5, 0.5, 0.5],  # Medium notes
    'low_arousal': [1.0, 0.5, 1.0, 1.5],  # Slow notes
}

# Default feature groupings - can be customized by user
DEFAULT_FEATURE_MAPPING = {
    'valence': ['sentiment_polarity', 'emotion_hope', 'peace_appeal'],
    'arousal': ['emotion_anger', 'emotion_fear', 'style_urgency', 'moral_outrage'],
    'tension': ['hostility_confrontation', 'military_intensity', 'intent_provocation'],
    'complexity': ['assertion_strength', 'factual_speculative', 'intent_persuasion'],
}

# Module requires music21 to be installed
import music21


def normalize_emotion_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all emotion values are in the 0-1 range.

    Args:
        df: DataFrame with emotion values

    Returns:
        DataFrame with normalized emotion values

    >>> import pandas as pd
    >>> df = pd.DataFrame({'emotion1': [-1, 0, 2], 'emotion2': [0, 0.5, 1]})
    >>> normalized = normalize_emotion_data(df)
    >>> normalized['emotion1'].min() >= 0 and normalized['emotion1'].max() <= 1
    True
    """
    # Create a copy of the dataframe to avoid modifying the original
    normalized_df = df.copy()

    # Process each column except 'index' if it exists
    for column in normalized_df.columns:
        if column == 'index':
            continue

        # Check if values are already in 0-1 range
        min_val = normalized_df[column].min()
        max_val = normalized_df[column].max()

        # Only normalize if values are not already in 0-1 range
        if min_val < 0 or max_val > 1:
            normalized_df[column] = (normalized_df[column] - min_val) / (
                max_val - min_val
            )

    return normalized_df


def map_features(
    df: pd.DataFrame, feature_mapping: dict[str, list[str]] = None
) -> pd.DataFrame:
    """
    Map raw emotion features to musical dimensions (valence, arousal, etc.).

    Args:
        df: DataFrame with raw emotion features
        feature_mapping: Dictionary mapping musical dimensions to lists of feature names

    Returns:
        DataFrame with mapped musical dimensions
    """
    if feature_mapping is None:
        feature_mapping = DEFAULT_FEATURE_MAPPING

    # Create a new dataframe for the mapped features
    mapped_df = pd.DataFrame(index=df.index)

    # For each musical dimension, compute the average of available features
    for dimension, features in feature_mapping.items():
        # Find which features exist in our dataframe
        available_features = [f for f in features if f in df.columns]

        if available_features:
            # Calculate mean of available features
            mapped_df[dimension] = df[available_features].mean(axis=1)
        else:
            # Default value if none of the features exist
            mapped_df[dimension] = 0.5

    # Add index column if it exists in original dataframe
    if 'index' in df.columns:
        mapped_df['index'] = df['index']

    return mapped_df


def transpose_scale(base_scale: list[int], root: int = 0) -> list[int]:
    """
    Transpose a scale to a new root note.

    Args:
        base_scale: List of scale degrees (0-11)
        root: New root note (0-11, where 0=C, 1=C#, etc.)

    Returns:
        Transposed scale

    >>> transpose_scale([0, 2, 4, 5, 7, 9, 11], 2)  # Transpose C major to D major
    [2, 4, 6, 7, 9, 11, 1]
    """
    return [(note + root) % 12 for note in base_scale]


def get_note_name(pitch_value: int) -> str:
    """
    Convert a pitch value (0-11) to a note name.

    Args:
        pitch_value: Integer pitch value (0=C, 1=C#, etc.)

    Returns:
        Note name string

    >>> get_note_name(0)
    'C'
    >>> get_note_name(1)
    'C#'
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[pitch_value % 12]


def _parse_note_name(note: str) -> tuple[str, str]:
    """
    Parse a note name to separate the letter name and accidentals.

    Args:
        note: Note name string (e.g., 'C', 'F#', 'E-flat')

    Returns:
        Tuple of (base_note, accidental)
    """
    # Handle flat notation in root
    if '-flat' in note:
        base = note[0]
        accidental = 'b'
        return base, accidental
    elif '-' in note and len(note) > 1 and note[1] == '-':
        base = note[0]
        accidental = 'b'
        return base, accidental
    elif len(note) > 1 and note[1] == 'b':
        base = note[0]
        accidental = 'b'
        return base, accidental
    elif len(note) > 1 and note[1] == '#':
        base = note[0]
        accidental = '#'
        return base, accidental
    else:
        return note[0], ''


def note_to_pitch_value(note: str) -> int:
    """
    Convert a note name to a pitch value (0-11).

    Args:
        note: Note name string (e.g., 'C', 'F#', 'Eb')

    Returns:
        Integer pitch value

    >>> note_to_pitch_value('C')
    0
    >>> note_to_pitch_value('F#')
    6
    >>> note_to_pitch_value('Eb')
    3
    """
    # Music21 note names (for all 12 semitones)
    note_names = {
        'C': 0,
        'C#': 1,
        'D': 2,
        'D#': 3,
        'E': 4,
        'F': 5,
        'F#': 6,
        'G': 7,
        'G#': 8,
        'A': 9,
        'A#': 10,
        'B': 11,
    }

    # Flat equivalents
    flat_equivalents = {'Db': 1, 'Eb': 3, 'Gb': 6, 'Ab': 8, 'Bb': 10}

    base, accidental = _parse_note_name(note)

    if accidental == 'b':
        note_str = f"{base}b"
        if note_str in flat_equivalents:
            return flat_equivalents[note_str]
        else:
            # Handle other flats by subtracting 1 from the base note
            return (note_names[base] - 1) % 12
    elif accidental == '#':
        return (note_names[base] + 1) % 12
    else:
        return note_names[base]


def transpose_chord_progression(
    chord_progression: list[tuple[str, str]], semitones: int
) -> list[tuple[str, str]]:
    """
    Transpose a chord progression by a number of semitones.

    Args:
        chord_progression: List of (root, quality) tuples
        semitones: Number of semitones to transpose by

    Returns:
        Transposed chord progression
    """
    # Music21 note names (for all 12 semitones)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    transposed_progression = []
    for root, quality in chord_progression:
        # Get the pitch value, transpose it, and convert back to a note name
        pitch_value = note_to_pitch_value(root)
        new_pitch_value = (pitch_value + semitones) % 12
        new_root = note_names[new_pitch_value]

        # Add to the transposed progression
        transposed_progression.append((new_root, quality))

    return transposed_progression


def generate_chord_notes(
    root: str, quality: str, inversion: int = 0, base_octave: int = 3
) -> list[str]:
    """
    Generate the notes for a chord based on root, quality, and inversion.

    Args:
        root: Root note name (e.g., 'C', 'F#')
        quality: Chord quality (e.g., 'major', 'minor', 'major-seventh')
        inversion: Chord inversion (0 = root position, 1 = first inversion, etc.)
        base_octave: Base octave for the chord

    Returns:
        List of note names in the chord
    """
    # Handle flat notation in root for music21 compatibility
    if '-flat' in root:
        root = root.replace('-flat', 'b')
    if '-' in root and len(root) > 1 and root[1] == '-':
        root = root[0] + 'b' + root[2:]

    # Define chord structures with semitone intervals from root
    chord_structures = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'diminished': [0, 3, 6],
        'augmented': [0, 4, 8],
        'suspended-fourth': [0, 5, 7],
        'suspended-second': [0, 2, 7],
        'major-seventh': [0, 4, 7, 11],
        'minor-seventh': [0, 3, 7, 10],
        'dominant-seventh': [0, 4, 7, 10],
        'half-diminished-seventh': [0, 3, 6, 10],
        'diminished-seventh': [0, 3, 6, 9],
        'augmented-seventh': [0, 4, 8, 10],
        'major-sixth': [0, 4, 7, 9],
        'minor-sixth': [0, 3, 7, 9],
        'minor-major-seventh': [0, 3, 7, 11],
        'add9': [0, 4, 7, 14],
        'add11': [0, 4, 7, 17],
        'ninth': [0, 4, 7, 10, 14],
        'minor-ninth': [0, 3, 7, 10, 14],
        'major-ninth': [0, 4, 7, 11, 14],
        'eleventh': [0, 4, 7, 10, 14, 17],
        'thirteenth': [0, 4, 7, 10, 14, 17, 21],
    }

    # Use the default major triad if quality not recognized
    intervals = chord_structures.get(quality, chord_structures['major'])

    # Create a music21 note for the root
    root_note = music21.note.Note(root + str(base_octave))

    # Generate notes based on the intervals
    notes = []
    for interval in intervals:
        note = music21.note.Note(root_note.pitch)
        note.pitch.transpose(interval, inPlace=True)
        notes.append(note.nameWithOctave)

    # Handle inversions
    if inversion > 0 and inversion < len(notes):
        # Move the bottom notes up an octave for the requested inversion
        for i in range(inversion):
            # Parse the note name and octave
            note_name = notes[i]
            base_name = ''.join([c for c in note_name if not c.isdigit()])
            octave = int(''.join([c for c in note_name if c.isdigit()]))

            # Move up an octave
            notes[i] = base_name + str(octave + 1)

    return notes


# Extended chord progressions library with varied emotional qualities
EXTENDED_CHORD_MAPPINGS = {
    # More complex chord progressions
    'complex': [
        [
            ('C', 'major-seventh'),
            ('A', 'minor-seventh'),
            ('F', 'major-seventh'),
            ('D', 'minor-seventh'),
        ],
        [
            ('C', 'major-ninth'),
            ('A', 'minor-ninth'),
            ('F', 'major-ninth'),
            ('G', 'dominant-seventh'),
        ],
        [
            ('C', 'major-seventh'),
            ('F', 'major-seventh'),
            ('D', 'minor-seventh'),
            ('G', 'suspended-fourth'),
        ],
        [('C', 'major-add9'), ('A', 'minor-add9'), ('F', 'major-add9'), ('G', 'ninth')],
    ],
    # Tense chord progressions
    'tense': [
        [
            ('C', 'minor'),
            ('G', 'dominant-seventh'),
            ('A', 'diminished'),
            ('D', 'half-diminished-seventh'),
        ],
        [
            ('C', 'minor-seventh'),
            ('F', 'minor'),
            ('G', 'dominant-seventh'),
            ('C', 'diminished'),
        ],
        [
            ('C', 'diminished'),
            ('D', 'half-diminished-seventh'),
            ('G', 'dominant-seventh'),
            ('C', 'minor'),
        ],
        [
            ('C', 'minor-major-seventh'),
            ('A', 'diminished-seventh'),
            ('D', 'minor-seventh'),
            ('G', 'altered'),
        ],
    ],
    # Positive chord progressions
    'positive': [
        [('C', 'major'), ('G', 'major'), ('A', 'minor'), ('F', 'major')],
        [('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')],
        [('C', 'major'), ('A', 'minor'), ('F', 'major'), ('G', 'dominant-seventh')],
        [
            ('C', 'major-sixth'),
            ('A', 'minor-seventh'),
            ('D', 'minor-seventh'),
            ('G', 'major'),
        ],
    ],
    # Negative chord progressions
    'negative': [
        [('C', 'minor'), ('G', 'minor'), ('E-flat', 'major'), ('F', 'minor')],
        [('C', 'minor'), ('F', 'minor'), ('G', 'minor'), ('C', 'minor')],
        [('C', 'minor'), ('A', 'diminished'), ('F', 'minor'), ('G', 'minor')],
        [
            ('C', 'minor-seventh'),
            ('F', 'minor-seventh'),
            ('G', 'dominant-seventh'),
            ('C', 'minor'),
        ],
    ],
    # Dreamy/ambiguous chord progressions
    'dreamy': [
        [
            ('C', 'major-seventh'),
            ('A', 'minor-seventh'),
            ('F', 'major-seventh'),
            ('F#', 'diminished'),
        ],
        [
            ('C', 'suspended-fourth'),
            ('G', 'suspended-second'),
            ('F', 'major-add9'),
            ('E', 'minor-seventh'),
        ],
        [
            ('C', 'major-seventh'),
            ('E', 'minor-seventh'),
            ('A', 'minor-seventh'),
            ('F', 'major-seventh'),
        ],
        [
            ('D', 'minor-ninth'),
            ('G', 'suspended-fourth'),
            ('C', 'major-add9'),
            ('A', 'minor-seventh'),
        ],
    ],
    # Triumphant chord progressions
    'triumphant': [
        [('C', 'major'), ('G', 'major'), ('F', 'major'), ('C', 'major')],
        [('C', 'major'), ('E', 'minor'), ('F', 'major'), ('G', 'major')],
        [('C', 'major'), ('D', 'major'), ('G', 'major'), ('C', 'major')],
        [('F', 'major'), ('C', 'major'), ('G', 'major'), ('C', 'major')],
    ],
    # Mysterious chord progressions
    'mysterious': [
        [
            ('C', 'minor-major-seventh'),
            ('A-flat', 'major-seventh'),
            ('F', 'minor-sixth'),
            ('G', 'dominant-seventh'),
        ],
        [
            ('C', 'minor-sixth'),
            ('E-flat', 'major-seventh'),
            ('A-flat', 'major-ninth'),
            ('G', 'altered'),
        ],
        [
            ('C', 'minor-seventh'),
            ('F', 'minor-ninth'),
            ('D', 'half-diminished-seventh'),
            ('G', 'dominant-ninth'),
        ],
        [
            ('C', 'suspended-second'),
            ('B-flat', 'major-add9'),
            ('A-flat', 'major-seventh'),
            ('G', 'suspended-fourth'),
        ],
    ],
}

# Define musical scales with emotional associations
EXTENDED_SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],  # C D E F G A B - bright, happy
    'natural_minor': [0, 2, 3, 5, 7, 8, 10],  # C D Eb F G Ab Bb - sad, pensive
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],  # C D Eb F G Ab B - exotic, mysterious
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],  # C D Eb F G A B - melancholic, expressive
    'dorian': [0, 2, 3, 5, 7, 9, 10],  # C D Eb F G A Bb - jazzy, contemplative
    'phrygian': [0, 1, 3, 5, 7, 8, 10],  # C Db Eb F G Ab Bb - exotic, tense
    'lydian': [0, 2, 4, 6, 7, 9, 11],  # C D E F# G A B - dreamy, futuristic
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],  # C D E F G A Bb - playful, bluesy
    'locrian': [0, 1, 3, 5, 6, 8, 10],  # C Db Eb F Gb Ab Bb - dissonant, unstable
    'pentatonic_major': [0, 2, 4, 7, 9],  # C D E G A - simple, folk
    'pentatonic_minor': [0, 3, 5, 7, 10],  # C Eb F G Bb - blues, eastern
    'blues': [0, 3, 5, 6, 7, 10],  # C Eb F F# G Bb - melancholic, soulful
    'whole_tone': [0, 2, 4, 6, 8, 10],  # C D E F# G# A# - dreamy, floating
    'chromatic': list(range(12)),  # All 12 tones - chaotic, complex
    'octatonic': [0, 2, 3, 5, 6, 8, 9, 11],  # C D Eb F Gb Ab A B - mysterious, jazz
}

# Extended duration patterns for increased rhythmic variety
EXTENDED_DURATION_MAPPINGS = {
    'high_arousal': [
        [0.25, 0.25, 0.5, 0.25, 0.25, 0.5],  # Fast notes
        [0.125, 0.125, 0.25, 0.125, 0.125, 0.25, 0.5],  # Very fast notes
        [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],  # Staccato feel
        [0.125, 0.125, 0.125, 0.125, 0.25, 0.25, 0.25, 0.25],  # Rapid notes
        [0.25, 0.125, 0.125, 0.25, 0.25, 0.5],  # Syncopated fast
    ],
    'medium_arousal': [
        [0.5, 0.5, 0.5, 0.5],  # Medium notes
        [0.25, 0.25, 0.5, 0.5, 0.5],  # Medium with some fast notes
        [0.5, 0.25, 0.25, 0.5, 0.5],  # Slightly syncopated
        [0.75, 0.25, 0.5, 0.5],  # Dotted rhythm
        [0.5, 0.5, 0.25, 0.25, 0.5],  # Mixed medium rhythm
    ],
    'low_arousal': [
        [1.0, 0.5, 1.0, 1.5],  # Slow notes
        [1.5, 0.5, 2.0],  # Very slow notes
        [1.0, 1.0, 1.0, 1.0],  # Steady slow notes
        [2.0, 0.5, 0.5, 1.0],  # Long then short
        [1.0, 0.5, 0.5, 2.0],  # Building to long note
    ],
}


def select_scale_for_emotion(valence: float, tension: float, complexity: float) -> str:
    """
    Select an appropriate scale type based on emotional parameters.

    Args:
        valence: Valence parameter (0-1)
        tension: Tension parameter (0-1)
        complexity: Complexity parameter (0-1)

    Returns:
        Scale type name from EXTENDED_SCALES
    """
    # High complexity favors more complex scales
    if complexity > 0.8:
        if tension > 0.7:
            return 'octatonic'
        elif valence > 0.6:
            return 'lydian'
        else:
            return 'harmonic_minor'

    # High tension favors more dissonant or unusual scales
    elif tension > 0.7:
        if valence < 0.3:
            return 'phrygian'
        elif complexity > 0.5:
            return 'locrian'
        else:
            return 'blues'

    # High valence favors more consonant, "happy" scales
    elif valence > 0.7:
        if complexity > 0.6:
            return 'mixolydian'
        else:
            return 'major'

    # Low valence favors more melancholic scales
    elif valence < 0.3:
        if complexity > 0.6:
            return 'melodic_minor'
        else:
            return 'natural_minor'

    # Medium values use more balanced scales
    else:
        if complexity > 0.6:
            return 'dorian'
        elif tension > 0.5:
            return 'pentatonic_minor'
        else:
            return 'pentatonic_major'


def select_chord_progression(
    valence: float,
    tension: float,
    complexity: float,
    chord_mappings: dict[str, list[list[tuple[str, str]]]] = None,
    scale_root: int = 0,
) -> list[tuple[str, str]]:
    """
    Select an appropriate chord progression based on emotional parameters.

    Args:
        valence: Valence parameter (0-1)
        tension: Tension parameter (0-1)
        complexity: Complexity parameter (0-1)
        chord_mappings: Dictionary mapping emotional states to chord progressions
        scale_root: Root note of the scale (0-11, where 0=C, 1=C#, etc.)

    Returns:
        List of (root, quality) tuples representing the chord progression
    """
    if chord_mappings is None:
        chord_mappings = EXTENDED_CHORD_MAPPINGS

    # Select the chord progression type based on emotional state
    if complexity > 0.7:
        progression_type = 'complex'
    elif tension > 0.7:
        progression_type = 'tense'
    elif valence > 0.7:
        progression_type = 'triumphant'
    elif valence > 0.5:
        progression_type = 'positive'
    elif valence < 0.3:
        progression_type = 'negative'
    elif tension > 0.5 and complexity > 0.5:
        progression_type = 'mysterious'
    else:
        progression_type = 'dreamy'

    # Select a random progression from the chosen type
    if progression_type in chord_mappings:
        progressions = chord_mappings[progression_type]
        if progressions:
            # Use emotional values to influence the selection rather than pure random
            # This creates more consistent progressions for similar emotional states
            emotion_seed = int((valence * 1000) + (tension * 100) + (complexity * 10))
            random.seed(emotion_seed)
            progression = progressions[emotion_seed % len(progressions)]
        else:
            # Fallback to first progression in 'positive' if empty
            progression = chord_mappings['positive'][0]
    else:
        # Fallback to first progression in 'positive' if type not found
        progression = chord_mappings['positive'][0]

    # Transpose the progression if needed
    if scale_root != 0:
        progression = transpose_chord_progression(progression, scale_root)

    return progression


def generate_markov_melody(
    scale: list[int],
    num_notes: int,
    valence: float,
    tension: float,
    complexity: float,
    seed_degree: int = None,
) -> list[int]:
    """
    Generate a melody using a Markov chain approach based on emotional parameters.

    Args:
        scale: List of scale degrees (0-11)
        num_notes: Number of notes to generate
        valence: Valence parameter (0-1)
        tension: Tension parameter (0-1)
        complexity: Complexity parameter (0-1)
        seed_degree: Starting scale degree (index in scale)

    Returns:
        List of scale degrees for the melody
    """
    # Define transition probabilities based on emotional parameters

    # For high valence (happy), favor upward movement and consonant intervals
    if valence > 0.7:
        # Movement probabilities: Down 2, Down 1, Same, Up 1, Up 2
        step_probs = [0.1, 0.2, 0.2, 0.3, 0.2]
        # Probability of leap (jumping more than 2 scale degrees)
        leap_prob = 0.2
        # Scale degrees to emphasize (0-indexed within scale)
        emphasis = [0, 2, 4]  # Root, 3rd, 5th

    # For low valence (sad), favor downward movement
    elif valence < 0.4:
        step_probs = [0.3, 0.3, 0.2, 0.1, 0.1]
        leap_prob = 0.15
        emphasis = [1, 3, 5]  # 2nd, 4th, 6th

    # For high tension, favor dissonant intervals
    elif tension > 0.7:
        step_probs = [0.2, 0.2, 0.1, 0.2, 0.3]
        leap_prob = 0.3
        emphasis = [1, 4, 6]  # 2nd, 5th, 7th

    # For high complexity, more varied movement
    elif complexity > 0.7:
        step_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        leap_prob = 0.4
        emphasis = [0, 2, 4, 6]  # Root, 3rd, 5th, 7th

    # Default/balanced
    else:
        step_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        leap_prob = 0.2
        emphasis = [0, 2, 4]  # Root, 3rd, 5th

    # Initialize the melody
    melody = []

    # Set the seed note
    if seed_degree is None:
        # Choose a seed note with emphasis on the selected degrees
        scale_idx = random.choices(
            range(len(scale)),
            weights=[
                1.5 if i % len(scale) in emphasis else 1.0 for i in range(len(scale))
            ],
            k=1,
        )[0]
    else:
        scale_idx = seed_degree % len(scale)

    melody.append(scale_idx)

    # Generate subsequent notes
    for _ in range(1, num_notes):
        current_degree = melody[-1]

        # Decide whether to make a leap or step
        if random.random() < leap_prob:
            # Make a leap (3+ scale degrees)
            leap_range = [-4, -3, 3, 4]
            leap = random.choice(leap_range)
            new_degree = (current_degree + leap) % len(scale)
        else:
            # Make a step (-2 to +2 scale degrees)
            step = random.choices([-2, -1, 0, 1, 2], weights=step_probs, k=1)[0]
            new_degree = (current_degree + step) % len(scale)

        # Apply emphasis on certain scale degrees occasionally
        if random.random() < 0.3:  # 30% chance to apply emphasis
            if new_degree % len(scale) not in emphasis:
                # Move to a nearby emphasized degree
                emphasized_degrees = [d % len(scale) for d in emphasis]
                emphasized_degrees.sort(key=lambda d: abs(d - new_degree))
                new_degree = emphasized_degrees[0]

        melody.append(new_degree)

    # Translate scale indices to actual pitches in the scale
    return [scale[idx % len(scale)] for idx in melody]


def generate_contoured_melody(
    scale: list[int], num_notes: int, valence: float, tension: float, complexity: float
) -> list[int]:
    """
    Generate a melody with a specific contour based on emotional parameters.

    Args:
        scale: List of scale degrees (0-11)
        num_notes: Number of notes to generate
        valence: Valence parameter (0-1)
        tension: Tension parameter (0-1)
        complexity: Complexity parameter (0-1)

    Returns:
        List of pitches for the melody
    """
    # Define melodic contours based on emotional parameters
    contours = {
        'ascending': lambda x: x / (num_notes - 1) if num_notes > 1 else 0.5,
        'descending': lambda x: 1 - (x / (num_notes - 1)) if num_notes > 1 else 0.5,
        'arch': lambda x: (
            4 * ((x / (num_notes - 1)) * (1 - (x / (num_notes - 1))))
            if num_notes > 1
            else 0.5
        ),
        'varch': lambda x: (
            1 - 4 * ((x / (num_notes - 1)) * (1 - (x / (num_notes - 1))))
            if num_notes > 1
            else 0.5
        ),
        'random': lambda x: random.random(),
        'wave': lambda x: (
            0.5 + 0.5 * math.sin(2 * math.pi * x / (num_notes / 2))
            if num_notes > 1
            else 0.5
        ),
    }

    # Select contour based on emotional parameters
    if valence > 0.7:
        if tension < 0.3:
            contour_func = contours['ascending']  # Happy, optimistic
        else:
            contour_func = contours['arch']  # Excited but tense
    elif valence < 0.3:
        if tension > 0.7:
            contour_func = contours['varch']  # Sad and tense
        else:
            contour_func = contours['descending']  # Sad, resigned
    else:
        if complexity > 0.7:
            contour_func = contours['random']  # Complex, unpredictable
        elif tension > 0.7:
            contour_func = contours['wave']  # Tense, fluctuating
        else:
            contour_func = contours['wave']  # Balanced, flowing

    # Generate melody based on contour
    melody = []
    scale_len = len(scale)
    for i in range(num_notes):
        # Calculate contour value (0-1) for this position
        contour_val = contour_func(i)

        # Map contour value to scale degree
        scale_idx = int(contour_val * scale_len) % scale_len

        # Add randomness based on complexity
        if random.random() < complexity:
            scale_idx = (scale_idx + random.choice([-1, 1])) % scale_len

        # Add the scale degree to melody
        melody.append(scale[scale_idx])

    return melody


def select_duration_pattern(
    arousal: float, duration_mappings: dict[str, list[list[float]]] = None
) -> list[float]:
    """
    Select an appropriate note duration pattern based on arousal level.

    Args:
        arousal: Arousal parameter (0-1)
        duration_mappings: Dictionary mapping arousal levels to lists of duration patterns

    Returns:
        List of note durations
    """
    if duration_mappings is None:
        duration_mappings = EXTENDED_DURATION_MAPPINGS

    # Select the category based on arousal
    if arousal > 0.7:
        category = 'high_arousal'
    elif arousal < 0.4:
        category = 'low_arousal'
    else:
        category = 'medium_arousal'

    # Get patterns for the category
    patterns = duration_mappings.get(category, duration_mappings['medium_arousal'])

    # Seed with arousal to get consistent results for similar values
    random.seed(int(arousal * 1000))

    # Select a random pattern from the category
    return random.choice(patterns)


def create_emotion_music(
    df: pd.DataFrame,
    output_file: str = "emotion_music.mid",
    scale: list[int] = None,
    scale_root: int = DEFAULT_SCALE_ROOT,
    force_constant_scale: bool = False,
    feature_mapping: dict[str, list[str]] = None,
    chord_mappings: dict[str, list[list[tuple[str, str]]]] = None,
    phrase_duration: float = DEFAULT_PHRASE_DURATION,
    tempo_range: tuple[int, int] = DEFAULT_TEMPO_RANGE,
    octave_range: tuple[int, int] = DEFAULT_OCTAVE_RANGE,
    velocity_range: tuple[int, int] = DEFAULT_VELOCITY_RANGE,
    duration_mappings: dict[str, list[list[float]]] = None,
    normalize: bool = True,
    use_chord_aware_melody: bool = True,
    progress_callback: Callable[[int, int], None] = None,
) -> str:
    """
    Create MIDI music based on emotion data with enhanced melodic and harmonic variation.

    Args:
        df: DataFrame with emotion parameters
        output_file: Path for output MIDI file
        scale: Scale to use for the piece (list of integers 0-11)
        scale_root: Root note of the scale (0-11, where 0=C, 1=C#, etc.)
        force_constant_scale: If True, use the provided scale throughout (no dynamic scale changes)
        feature_mapping: Dictionary mapping musical dimensions to lists of feature names
        chord_mappings: Dictionary mapping emotional states to chord progressions
        phrase_duration: Duration of each emotion segment in beats
        tempo_range: Range of tempos to map arousal to (min, max)
        octave_range: Range of octaves for melody (min, max)
        velocity_range: Range of MIDI velocities for dynamics (min, max)
        duration_mappings: Dictionary mapping arousal levels to note durations
        normalize: Whether to normalize emotion values to 0-1 range
        use_chord_aware_melody: Whether to use chord-aware melody generation
        progress_callback: Function to call with progress updates (current, total)

    Returns:
        Path to the generated MIDI file
    """
    # Import math for some of the contour functions
    import math

    # Set defaults
    if scale is None:
        scale = DEFAULT_SCALE
    if feature_mapping is None:
        feature_mapping = DEFAULT_FEATURE_MAPPING
    if chord_mappings is None:
        chord_mappings = EXTENDED_CHORD_MAPPINGS
    if duration_mappings is None:
        duration_mappings = EXTENDED_DURATION_MAPPINGS

    # Prepare the data
    if normalize:
        df = normalize_emotion_data(df)

    # Map raw features to musical dimensions
    mapped_df = map_features(df, feature_mapping)

    print(f"Creating music from {len(mapped_df)} emotion data points")

    # Create a music21 score
    score = music21.stream.Score()

    # Add metadata
    metadata = music21.metadata.Metadata()
    metadata.title = "Emotion Sonification"
    metadata.composer = "AI Composer"
    score.insert(0, metadata)

    # Create parts for melody and accompaniment
    melody_part = music21.stream.Part()
    melody_part.insert(0, music21.instrument.Piano())

    accomp_part = music21.stream.Part()
    accomp_part.insert(0, music21.instrument.Piano())

    # Add a counter melody part
    counter_melody_part = music21.stream.Part()
    counter_melody_part.insert(0, music21.instrument.Piano())

    # Current position in the score
    current_offset = 0.0

    # If using constant scale, create music21 scale object once
    if force_constant_scale:
        constant_scale = transpose_scale(scale, scale_root)
        # Create music21 scale object for note generation
        constant_scale_pitches = [music21.pitch.Pitch(n % 12) for n in constant_scale]
        constant_concrete_scale = music21.scale.ConcreteScale(
            pitches=constant_scale_pitches
        )

    # Process each row in the dataframe
    total_rows = len(mapped_df)
    last_melody_note = None  # Track last melody note for smoother transitions

    for idx, row in mapped_df.iterrows():
        # Update progress if callback provided
        if progress_callback and idx % max(1, total_rows // 10) == 0:
            progress_callback(idx, total_rows)

        # Extract emotional dimensions
        valence = row.get('valence', 0.5)  # positive/negative sentiment
        arousal = row.get('arousal', 0.5)  # energy/intensity
        tension = row.get('tension', 0.5)  # conflict/dissonance
        complexity = row.get('complexity', 0.5)  # musical complexity

        # 1. Determine scale to use
        if force_constant_scale:
            # Use the provided scale throughout
            current_scale = constant_scale
            concrete_scale = constant_concrete_scale
        else:
            # Select scale type based on emotional state
            scale_type = select_scale_for_emotion(valence, tension, complexity)
            current_scale_degrees = EXTENDED_SCALES[scale_type]

            # Apply root transposition
            current_scale = transpose_scale(current_scale_degrees, scale_root)

            # Create music21 scale object for note generation
            scale_pitches = [music21.pitch.Pitch(n % 12) for n in current_scale]
            concrete_scale = music21.scale.ConcreteScale(pitches=scale_pitches)

        # 2. Determine tempo from arousal
        tempo_min, tempo_max = tempo_range
        tempo = int(tempo_min + arousal * (tempo_max - tempo_min))

        # Add tempo marking
        tempo_mark = music21.tempo.MetronomeMark(number=tempo)
        melody_part.insert(current_offset, tempo_mark)

        # 3. Get chord progression based on emotional state
        chord_progression = select_chord_progression(
            valence, tension, complexity, chord_mappings, scale_root
        )

        # 4. Create a musical phrase
        phrase_offset = current_offset

        # Add chords to accompaniment
        for i, (root, quality) in enumerate(chord_progression):
            # Determine chord inversion for variety
            inversion = 0
            if complexity > 0.5:
                # More complex emotional states use inversions
                inversion = i % 3

            # Get chord notes with the determined inversion
            chord_notes = generate_chord_notes(root, quality, inversion)

            # Create chord
            chord = music21.chord.Chord(chord_notes)

            # Set chord duration
            chord.duration = music21.duration.Duration(
                phrase_duration / len(chord_progression)
            )

            # Set chord loudness based on arousal
            velocity_min, velocity_max = velocity_range
            chord_velocity = int(velocity_min + arousal * (velocity_max - velocity_min))
            chord.volume.velocity = min(127, chord_velocity)

            # Position chord in the accompaniment part
            chord_offset = phrase_offset + (
                i * phrase_duration / len(chord_progression)
            )
            accomp_part.insert(chord_offset, chord)

            # Add arpeggiated notes for high complexity
            if complexity > 0.6:
                # Create arpeggios for the chord
                arpeggio_durations = [0.25] * 4  # Default to 16th notes
                if arousal < 0.4:
                    arpeggio_durations = [0.5] * 2  # Slower arpeggios for low arousal

                for j, note_name in enumerate(chord_notes):
                    # Skip some notes based on complexity
                    if random.random() > complexity:
                        continue

                    # Create note
                    note = music21.note.Note(note_name)
                    note.duration = music21.duration.Duration(
                        arpeggio_durations[j % len(arpeggio_durations)]
                    )

                    # Set velocity
                    note.volume.velocity = min(115, chord_velocity - 10)

                    # Position in accompaniment part
                    note_offset = chord_offset + (
                        j * sum(arpeggio_durations[:1]) / len(chord_notes)
                    )

                    # Only add if we're still within the chord's time span
                    if note_offset < chord_offset + chord.duration.quarterLength:
                        accomp_part.insert(note_offset, note)

        # 5. Create melody
        num_notes = max(
            4, int(8 * (1 + complexity))
        )  # More notes for higher complexity

        if use_chord_aware_melody:
            # Use chord-aware melody generation
            notes_per_chord = max(2, int((num_notes / len(chord_progression)) + 0.5))
            melody_pitches, durations = generate_chord_aware_melody(
                current_scale,
                chord_progression,
                notes_per_chord,
                valence,
                tension,
                complexity,
                phrase_duration,
                last_melody_note,
            )

            # Remember last note for continuity
            if melody_pitches:
                last_melody_note = melody_pitches[-1]

            # Place melody notes
            total_duration = 0
            note_idx = 0

            while note_idx < len(melody_pitches) and total_duration < phrase_duration:
                # Get note
                pitch = melody_pitches[note_idx]

                # Get note duration
                if note_idx < len(durations):
                    duration = durations[note_idx]
                else:
                    duration = 0.5  # Default

                # Ensure we don't exceed phrase duration
                if total_duration + duration > phrase_duration:
                    duration = phrase_duration - total_duration
                    if duration <= 0:
                        break

                # Create note from pitch value
                note = music21.note.Note()
                note.pitch = music21.pitch.Pitch(pitch)

                # Adjust octave based on position in current scale and melody
                octave_min, octave_max = octave_range
                base_octave = octave_min + 1

                # Get previous note's octave if available
                prev_octave = None
                if note_idx > 0 and note_idx - 1 < len(melody_part):
                    try:
                        prev_elements = melody_part.getElementsByOffset(
                            phrase_offset + total_duration - 0.001,
                            phrase_offset + total_duration,
                        )
                        if prev_elements and isinstance(
                            prev_elements[0], music21.note.Note
                        ):
                            prev_octave = prev_elements[0].pitch.octave
                    except:
                        pass

                if prev_octave is not None:
                    # Choose octave for smooth voice leading
                    # Find the interval between the previous pitch class and this one
                    prev_pitch_class = melody_pitches[note_idx - 1] % 12
                    current_pitch_class = pitch % 12

                    # Determine if we should go up or down an octave
                    interval = (current_pitch_class - prev_pitch_class) % 12
                    if interval > 6 and current_pitch_class < prev_pitch_class:
                        # Large upward interval might be better as downward
                        note.pitch.octave = prev_octave - 1
                    elif interval < 6 and current_pitch_class > prev_pitch_class:
                        # Small downward interval should stay in same octave
                        note.pitch.octave = prev_octave
                    else:
                        # Default to same octave
                        note.pitch.octave = prev_octave
                else:
                    # No previous note, use base octave
                    note.pitch.octave = base_octave

                # Ensure octave is within range
                note.pitch.octave = max(octave_min, min(octave_max, note.pitch.octave))

                # Set duration
                note.duration = music21.duration.Duration(duration)

                # Set velocity based on arousal and position
                velocity_min, velocity_max = velocity_range
                note_velocity = int(
                    velocity_min + arousal * (velocity_max - velocity_min)
                )

                # Emphasize important beats
                beat_position = (total_duration % 4) / 4.0
                if beat_position < 0.25:  # Strong beat
                    note_velocity += 10

                # Apply musical dynamics
                phrase_position = total_duration / phrase_duration
                if phrase_position < 0.3:
                    # Start slightly quieter
                    note_velocity -= 5
                elif 0.3 <= phrase_position < 0.7:
                    # Middle is louder
                    note_velocity += 5
                else:
                    # End tapers off
                    note_velocity -= int(10 * (phrase_position - 0.7) / 0.3)

                note.volume.velocity = min(127, max(30, note_velocity))

                # Position note in the melody part
                note_offset = phrase_offset + total_duration
                melody_part.insert(note_offset, note)

                total_duration += duration
                note_idx += 1

        else:
            # Use previous melody generation approaches based on complexity
            # Generate melody using different strategies based on complexity
            if complexity > 0.7:
                # Use Markov chain for complex melodies
                melody_pitches = generate_markov_melody(
                    current_scale, num_notes, valence, tension, complexity, seed_note
                )
            else:
                # Use contoured melody for simpler, more predictable patterns
                melody_pitches = generate_contoured_melody(
                    current_scale, num_notes, valence, tension, complexity
                )

            # Remember last note for continuity
            if melody_pitches:
                last_melody_note = melody_pitches[-1]

            # Create rhythm pattern based on arousal
            durations = select_duration_pattern(arousal, duration_mappings)

            # Place melody notes
            total_duration = 0
            for i, pitch in enumerate(melody_pitches):
                # Get note duration
                duration = durations[i % len(durations)]

                # Adjust duration based on phrase length
                if total_duration + duration > phrase_duration:
                    duration = phrase_duration - total_duration
                    if duration <= 0:
                        break

                # Create note from pitch value
                note = music21.note.Note()
                note.pitch = music21.pitch.Pitch(pitch)

                # Adjust octave based on emotional parameters and position
                octave_min, octave_max = octave_range
                base_octave = octave_min + 1

                # Adjust octave based on valence and position in phrase
                if valence > 0.7:
                    # Higher octave for positive emotions, especially at climactic points
                    octave_adj = 1 if i == num_notes // 2 else 0
                elif valence < 0.3 and i > num_notes // 2:
                    # Lower octave for negative emotions, especially towards the end
                    octave_adj = -1
                else:
                    octave_adj = 0

                # Apply octave adjustment with bounds checking
                note.pitch.octave = max(
                    octave_min, min(octave_max, base_octave + octave_adj)
                )

                # Set duration
                note.duration = music21.duration.Duration(duration)

                # Set velocity based on arousal and position
                velocity_min, velocity_max = velocity_range
                note_velocity = int(
                    velocity_min + arousal * (velocity_max - velocity_min)
                )

                # Emphasize important beats
                note_velocity += 10 if i % 4 == 0 else 0

                # Adjust for musical dynamics (crescendo/diminuendo)
                if i < num_notes // 2:
                    # Crescendo in first half
                    position_factor = i / (num_notes // 2)
                    note_velocity += int(10 * position_factor)
                else:
                    # Diminuendo in second half
                    position_factor = (i - num_notes // 2) / (
                        num_notes - num_notes // 2
                    )
                    note_velocity -= int(10 * position_factor)

                note.volume.velocity = min(127, max(30, note_velocity))

                # Position note in the melody part
                note_offset = phrase_offset + total_duration
                melody_part.insert(note_offset, note)

                total_duration += duration

        # 6. Add a counter-melody for high complexity
        if complexity > 0.5:
            # Create counter-melody using a different generation approach
            num_counter_notes = max(3, int(6 * complexity))

            if complexity > 0.7:
                # For high complexity, more independent counter-melody
                counter_pitches = generate_markov_melody(
                    current_scale, num_counter_notes, 1 - valence, tension, complexity
                )
            else:
                # For medium complexity, invert the main melody contour
                counter_pitches = generate_contoured_melody(
                    current_scale, num_counter_notes, 1 - valence, tension, complexity
                )

            # Create rhythm pattern for counter-melody (slightly offset)
            counter_durations = select_duration_pattern(
                max(0.3, arousal - 0.2), duration_mappings
            )

            # Place counter-melody notes
            counter_total_duration = 0
            counter_offset = phrase_offset + 0.25  # Start slightly after main melody

            for i, pitch in enumerate(counter_pitches):
                # Get note duration
                duration = counter_durations[i % len(counter_durations)]

                # Adjust duration based on phrase length
                if counter_total_duration + duration > phrase_duration - 0.25:
                    duration = phrase_duration - 0.25 - counter_total_duration
                    if duration <= 0:
                        break

                # Create note from pitch value
                note = music21.note.Note()
                note.pitch = music21.pitch.Pitch(pitch)

                # Set octave (typically lower than melody)
                note.pitch.octave = octave_range[0]

                # Set duration
                note.duration = music21.duration.Duration(duration)

                # Set velocity (softer than main melody)
                counter_velocity = int(
                    velocity_min + arousal * (velocity_max - velocity_min) * 0.8
                )
                note.volume.velocity = min(110, counter_velocity)

                # Position note in the counter-melody part
                note_offset = counter_offset + counter_total_duration
                counter_melody_part.insert(note_offset, note)

                counter_total_duration += duration

        # Move to next phrase
        current_offset += phrase_duration

    # Final progress update
    if progress_callback:
        progress_callback(total_rows, total_rows)

    # Add parts to the score
    score.insert(0, melody_part)
    score.insert(0, accomp_part)
    score.insert(0, counter_melody_part)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write to MIDI file
    score.write('midi', fp=output_file)
    print(f"MIDI file created: {output_file}")

    return output_file


def visualize_emotion_data(
    df: pd.DataFrame,
    output_file: str = "emotion_visualization.png",
    feature_mapping: dict[str, list[str]] = None,
    show_plot: bool = False,
    title: str = 'Emotion Parameters Over Time',
    figsize: tuple[int, int] = (12, 6),
    dpi: int = 300,
):
    """
    Create a visualization of the emotion data.

    Args:
        df: DataFrame with emotion parameters
        output_file: Path to save visualization
        feature_mapping: Dictionary mapping musical dimensions to lists of feature names
        show_plot: Whether to display the plot (in addition to saving)
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        dpi: DPI for the saved image
    """
    # If feature mapping provided, use mapped features
    if feature_mapping is not None:
        mapped_df = map_features(df, feature_mapping)
        key_features = mapped_df.columns
        plot_df = mapped_df
    else:
        # Otherwise select some default columns
        potential_features = [
            'sentiment_polarity',
            'emotion_anger',
            'emotion_fear',
            'emotion_hope',
            'hostility_confrontation',
            'peace_appeal',
        ]
        key_features = [col for col in potential_features if col in df.columns][
            :6
        ]  # Limit to 6
        plot_df = df

    # Remove 'index' from features to plot
    if 'index' in key_features:
        key_features = [f for f in key_features if f != 'index']

    if not any(key_features):
        print("No suitable emotion features found for visualization")
        return

    plt.figure(figsize=figsize)

    # Plot each feature
    for feature in key_features:
        plt.plot(plot_df.index, plot_df[feature], label=feature)

    plt.title(title)
    plt.xlabel('Time Index')
    plt.ylabel('Emotion Value')
    plt.legend()
    plt.grid(alpha=0.3)

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plt.savefig(output_file, dpi=dpi)
    print(f"Visualization saved to {output_file}")

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_musical_parameters(
    df: pd.DataFrame,
    output_file: str = "musical_parameters.png",
    feature_mapping: dict[str, list[str]] = None,
    show_plot: bool = False,
    figsize: tuple[int, int] = (14, 8),
    dpi: int = 300,
):
    """
    Create a visualization of how emotion data maps to musical parameters.

    Args:
        df: DataFrame with emotion parameters
        output_file: Path to save visualization
        feature_mapping: Dictionary mapping musical dimensions to lists of feature names
        show_plot: Whether to display the plot (in addition to saving)
        figsize: Figure size (width, height) in inches
        dpi: DPI for the saved image
    """
    # Map raw features to musical dimensions
    if feature_mapping is None:
        feature_mapping = DEFAULT_FEATURE_MAPPING

    mapped_df = map_features(df, feature_mapping)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Mapping of Emotions to Musical Parameters', fontsize=16)

    # 1. Valence -> Scale type and chord quality
    axs[0, 0].plot(mapped_df.index, mapped_df['valence'], 'b-')
    axs[0, 0].set_title('Valence → Scale & Chord Quality')
    axs[0, 0].set_ylabel('Valence Value')
    axs[0, 0].set_ylim(0, 1)

    # Add annotations for scale types
    valence_thresholds = [0.3, 0.7]
    valence_labels = ['Minor/Sad Scales', 'Neutral Scales', 'Major/Happy Scales']

    for i in range(len(valence_thresholds) + 1):
        if i == 0:
            y_pos = 0.15
        elif i == 1:
            y_pos = 0.5
        else:
            y_pos = 0.85

        axs[0, 0].text(
            len(mapped_df) * 0.95,
            y_pos,
            valence_labels[i],
            ha='right',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7),
        )

    # 2. Arousal -> Tempo and note duration
    axs[0, 1].plot(mapped_df.index, mapped_df['arousal'], 'r-')
    axs[0, 1].set_title('Arousal → Tempo & Note Duration')
    axs[0, 1].set_ylabel('Arousal Value')
    axs[0, 1].set_ylim(0, 1)

    # Add annotations for tempo
    tempo_min, tempo_max = DEFAULT_TEMPO_RANGE
    arousal_thresholds = [0.3, 0.7]

    for i, threshold in enumerate(arousal_thresholds):
        tempo = int(tempo_min + threshold * (tempo_max - tempo_min))
        y_pos = threshold
        axs[0, 1].axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
        axs[0, 1].text(
            len(mapped_df) * 0.95,
            y_pos,
            f'~{tempo} BPM',
            ha='right',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7),
        )

    # 3. Tension -> Harmony and dissonance
    axs[1, 0].plot(mapped_df.index, mapped_df['tension'], 'g-')
    axs[1, 0].set_title('Tension → Harmony & Dissonance')
    axs[1, 0].set_ylabel('Tension Value')
    axs[1, 0].set_ylim(0, 1)

    tension_thresholds = [0.3, 0.7]
    tension_labels = ['Consonant', 'Moderate Tension', 'Dissonant']

    for i in range(len(tension_thresholds) + 1):
        if i == 0:
            y_pos = 0.15
        elif i == 1:
            y_pos = 0.5
        else:
            y_pos = 0.85

        axs[1, 0].text(
            len(mapped_df) * 0.95,
            y_pos,
            tension_labels[i],
            ha='right',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7),
        )

    # 4. Complexity -> Musical structure and melodic variation
    axs[1, 1].plot(mapped_df.index, mapped_df['complexity'], 'm-')
    axs[1, 1].set_title('Complexity → Structure & Variation')
    axs[1, 1].set_ylabel('Complexity Value')
    axs[1, 1].set_ylim(0, 1)

    complexity_thresholds = [0.3, 0.7]
    complexity_labels = ['Simple', 'Moderate', 'Complex']

    for i in range(len(complexity_thresholds) + 1):
        if i == 0:
            y_pos = 0.15
        elif i == 1:
            y_pos = 0.5
        else:
            y_pos = 0.85

        axs[1, 1].text(
            len(mapped_df) * 0.95,
            y_pos,
            complexity_labels[i],
            ha='right',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7),
        )

    # Adjust layout and labels
    for ax in axs.flat:
        ax.set_xlabel('Time Index')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plt.savefig(output_file, dpi=dpi)
    print(f"Musical parameters visualization saved to {output_file}")

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_music_from_emotions(
    df: pd.DataFrame,
    output_dir: str = "emotion_music",
    output_filename: str = "emotion_music.mid",
    scale: list[int] = None,
    scale_root: int = DEFAULT_SCALE_ROOT,
    force_constant_scale: bool = False,
    feature_mapping: dict[str, list[str]] = None,
    chord_mappings: dict[str, list[list[tuple[str, str]]]] = None,
    phrase_duration: float = DEFAULT_PHRASE_DURATION,
    tempo_range: tuple[int, int] = DEFAULT_TEMPO_RANGE,
    octave_range: tuple[int, int] = DEFAULT_OCTAVE_RANGE,
    velocity_range: tuple[int, int] = DEFAULT_VELOCITY_RANGE,
    duration_mappings: dict[str, list[list[float]]] = None,
    visualize: bool = True,
    visualize_parameters: bool = True,
    normalize: bool = True,
    use_chord_aware_melody: bool = True,
) -> dict[str, str]:
    """
    Generate music and visualization from emotion data with enhanced musical variation.

    Args:
        df: DataFrame with emotion parameters
        output_dir: Directory to save output files
        output_filename: Filename for the MIDI output
        scale: Scale to use for the piece (list of integers 0-11)
        scale_root: Root note of the scale (0-11, where 0=C, 1=C#, etc.)
        force_constant_scale: If True, use the provided scale throughout (no dynamic scale changes)
        feature_mapping: Dictionary mapping musical dimensions to lists of feature names
        chord_mappings: Dictionary mapping emotional states to chord progressions
        phrase_duration: Duration of each emotion segment in beats
        tempo_range: Range of tempos to map arousal to (min, max)
        octave_range: Range of octaves for melody (min, max)
        velocity_range: Range of MIDI velocities for dynamics (min, max)
        duration_mappings: Dictionary mapping arousal levels to note durations
        visualize: Whether to create data visualization
        visualize_parameters: Whether to create musical parameters visualization
        normalize: Whether to normalize emotion values to 0-1 range
        use_chord_aware_melody: Whether to use chord-aware melody generation

    Returns:
        Dictionary with paths to output files
    """
    # Start timing
    start_time = time.time()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    output_files = {}

    # Define progress callback
    def progress_callback(current, total):
        print(
            f"Progress: {current}/{total} data points processed ({current/total*100:.1f}%)"
        )

    # Generate emotion data visualization
    if visualize:
        vis_file = os.path.join(output_dir, "emotion_visualization.png")
        visualize_emotion_data(df, vis_file, feature_mapping=feature_mapping)
        output_files['emotion_visualization'] = vis_file

    # Generate musical parameters visualization
    if visualize_parameters:
        params_file = os.path.join(output_dir, "musical_parameters.png")
        visualize_musical_parameters(df, params_file, feature_mapping=feature_mapping)
        output_files['musical_parameters'] = params_file

    # Generate music
    try:
        music_file = os.path.join(output_dir, output_filename)
        create_emotion_music(
            df,
            music_file,
            scale=scale,
            scale_root=scale_root,
            force_constant_scale=force_constant_scale,
            feature_mapping=feature_mapping,
            chord_mappings=chord_mappings,
            phrase_duration=phrase_duration,
            tempo_range=tempo_range,
            octave_range=octave_range,
            velocity_range=velocity_range,
            duration_mappings=duration_mappings,
            normalize=normalize,
            use_chord_aware_melody=use_chord_aware_melody,
            progress_callback=progress_callback,
        )
        output_files['music'] = music_file
    except Exception as e:
        print(f"Error generating music: {e}")
        import traceback

        traceback.print_exc()

    # Report execution time
    execution_time = time.time() - start_time
    print(f"Music generation completed in {execution_time:.2f} seconds")

    # Print summary
    print("\nGenerated files:")
    for file_type, file_path in output_files.items():
        print(f"  - {file_type}: {file_path}")

    return output_files


def generate_chord_aware_melody(
    scale: list[int],
    chord_progression: list[tuple[str, str]],
    num_notes_per_chord: int,
    valence: float,
    tension: float,
    complexity: float,
    phrase_duration: float,
    last_melody_note: int = None,
) -> tuple[list[int], list[float]]:
    """
    Generate a melody that is aware of the chord progression and maintains
    smooth voice leading.

    Args:
        scale: List of scale degrees (0-11)
        chord_progression: List of (root, quality) tuples
        num_notes_per_chord: Number of notes to generate per chord
        valence: Valence parameter (0-1)
        tension: Tension parameter (0-1)
        complexity: Complexity parameter (0-1)
        phrase_duration: Duration of the entire phrase in beats
        last_melody_note: The last note from the previous melody segment (if any)

    Returns:
        Tuple of (melody_pitches, durations)
    """
    melody_pitches = []

    # Calculate chord durations
    chord_duration = phrase_duration / len(chord_progression)

    # For rhythm generation, use complexity and arousal-like parameter derived from valence/tension
    effective_arousal = 0.3 + (0.4 * tension) + (0.3 * (1 - valence))

    # Select duration pattern based on this effective arousal
    durations = select_duration_pattern(effective_arousal)

    # Generate notes for each chord
    current_position = 0.0
    all_durations = []

    # Process each chord in the progression
    for chord_idx, (root, quality) in enumerate(chord_progression):
        # Convert root note to pitch value
        root_pitch = note_to_pitch_value(root)

        # Get the actual chord notes
        chord_note_names = generate_chord_notes(root, quality)
        chord_pitches = [
            note_to_pitch_value(note.split('/')[0]) for note in chord_note_names
        ]

        # Create pitch classes (0-11) for chord tones
        chord_pitch_classes = [p % 12 for p in chord_pitches]

        # Determine consonant and dissonant scale degrees relative to this chord
        scale_pitch_classes = [p % 12 for p in scale]

        # Define chord tones, stable tones, and unstable tones
        chord_tones = chord_pitch_classes
        stable_tones = []
        unstable_tones = []

        for pitch in scale_pitch_classes:
            # Skip pitches already categorized as chord tones
            if pitch in chord_tones:
                continue

            # Check the interval with the root
            interval = (pitch - root_pitch) % 12

            # Categorize based on consonance
            if interval in [
                0,
                7,
                5,
                4,
                8,
            ]:  # Perfect consonances and major/minor thirds/sixths
                stable_tones.append(pitch)
            else:
                unstable_tones.append(pitch)

        # Generate notes for this chord
        notes_for_this_chord = []
        chord_start_position = current_position

        while (
            current_position < chord_start_position + chord_duration
            and len(notes_for_this_chord) < num_notes_per_chord * 2
        ):
            # Determine if this note is on a strong beat
            is_strong_beat = (current_position - chord_start_position) % 1.0 < 0.1

            # Select pitch based on position and emotional context
            if len(notes_for_this_chord) == 0:
                # First note of chord - prefer chord tones
                if last_melody_note is not None and random.random() < 0.3:
                    # Sometimes continue smoothly from last note
                    pitch_options = chord_tones + stable_tones
                    # Find closest pitch to last note
                    last_pitch_class = last_melody_note % 12
                    pitch = min(
                        pitch_options,
                        key=lambda p: min(
                            (p - last_pitch_class) % 12, (last_pitch_class - p) % 12
                        ),
                    )
                else:
                    # Usually start with a chord tone
                    pitch = random.choice(chord_tones)
            elif is_strong_beat and random.random() < 0.7:
                # Strong beats often use chord tones
                pitch = random.choice(chord_tones)
            else:
                # Regular beats use a mix based on emotional context

                # Get the previous note
                prev_note = notes_for_this_chord[-1] % 12

                # Weights for different note types
                chord_weight = 0.5 - (0.3 * complexity) + (0.2 * valence)
                stable_weight = 0.3 + (0.1 * complexity) + (0.1 * valence)
                unstable_weight = (
                    0.2 + (0.2 * complexity) + (0.2 * tension) - (0.1 * valence)
                )

                # Normalize weights
                total_weight = chord_weight + stable_weight + unstable_weight
                chord_weight /= total_weight
                stable_weight /= total_weight
                unstable_weight /= total_weight

                # Select note type
                if not stable_tones and not unstable_tones:
                    # Fallback if no categorized tones
                    note_type = "chord"
                else:
                    # Choose note type based on weights
                    options = []
                    weights = []

                    if chord_tones:
                        options.append("chord")
                        weights.append(chord_weight)

                    if stable_tones:
                        options.append("stable")
                        weights.append(stable_weight)

                    if unstable_tones:
                        options.append("unstable")
                        weights.append(unstable_weight)

                    # Normalize weights again if needed
                    total = sum(weights)
                    if total > 0:
                        weights = [w / total for w in weights]
                    else:
                        weights = [1 / len(options)] * len(options)

                    note_type = random.choices(options, weights=weights, k=1)[0]

                # Select the actual pitch based on chosen type and voice leading
                if note_type == "chord":
                    pitch_options = chord_tones
                elif note_type == "stable":
                    pitch_options = stable_tones
                elif note_type == "unstable":
                    pitch_options = unstable_tones
                else:
                    # Fallback
                    pitch_options = scale_pitch_classes

                # Apply voice leading - prefer smaller intervals
                if pitch_options:
                    # Find distances to previous note
                    distances = [
                        min((p - prev_note) % 12, (prev_note - p) % 12)
                        for p in pitch_options
                    ]

                    # Weight by inverse distance (closer = higher weight)
                    # But avoid exact repeats sometimes based on complexity
                    repeat_penalty = (
                        0.5
                        if complexity > 0.5 and len(notes_for_this_chord) > 1
                        else 0.0
                    )

                    # Calculate weights
                    pitch_weights = []
                    for d in distances:
                        if d == 0:  # Same note
                            weight = 1.0 - repeat_penalty
                        elif d <= 2:  # Step (1-2 semitones)
                            weight = 0.8
                        elif d <= 4:  # Small leap (3-4 semitones)
                            weight = 0.6
                        else:  # Large leap
                            weight = 0.3
                        pitch_weights.append(weight)

                    # Normalize weights
                    if sum(pitch_weights) > 0:
                        pitch_weights = [w / sum(pitch_weights) for w in pitch_weights]
                    else:
                        pitch_weights = [1 / len(pitch_options)] * len(pitch_options)

                    # Choose pitch
                    pitch = random.choices(pitch_options, weights=pitch_weights, k=1)[0]
                else:
                    # Fallback if no options
                    pitch = random.choice(scale_pitch_classes)

            # Add the selected pitch to our melody
            notes_for_this_chord.append(pitch)

            # Get the duration for this note
            duration = durations[len(all_durations) % len(durations)]

            # Make sure we don't exceed chord duration
            if current_position + duration > chord_start_position + chord_duration:
                duration = chord_start_position + chord_duration - current_position

            # Add the duration
            all_durations.append(duration)

            # Update position
            current_position += duration

        # Add this chord's notes to the overall melody
        melody_pitches.extend(notes_for_this_chord)

        # Update last melody note for the next chord
        if notes_for_this_chord:
            last_melody_note = notes_for_this_chord[-1]

    # Return both the pitches and their durations
    return melody_pitches, all_durations


# Example functions


def create_sample_dataframe(number_of_pts: int = 20) -> pd.DataFrame:
    """
    Create a sample dataframe with emotion parameters.

    Args:
        number_of_pts: Number of data points to generate

    Returns:
        DataFrame with synthetic emotion parameters
    """
    sample_data = {
        'sentiment_polarity': np.linspace(0.3, 0.8, number_of_pts),
        'emotion_anger': np.linspace(0.8, 0.2, number_of_pts),
        'emotion_fear': np.cos(np.linspace(0, 2 * np.pi, number_of_pts)) * 0.2 + 0.5,
        'emotion_hope': np.sin(np.linspace(0, 2 * np.pi, number_of_pts)) * 0.3 + 0.6,
        'style_urgency': np.sin(np.linspace(0, np.pi, number_of_pts)) * 0.4 + 0.5,
        'hostility_confrontation': np.abs(
            np.sin(np.linspace(0, 3 * np.pi, number_of_pts))
        )
        * 0.5
        + 0.3,
        'peace_appeal': np.cos(np.linspace(0, 3 * np.pi, number_of_pts)) * 0.3 + 0.6,
    }

    return pd.DataFrame(sample_data)


def create_dynamic_emotion_dataframe(number_of_pts: int = 30) -> pd.DataFrame:
    """
    Create a sample dataframe with more dramatic emotional changes.

    Args:
        number_of_pts: Number of data points to generate

    Returns:
        DataFrame with synthetic emotion parameters featuring dramatic shifts
    """
    # Create segments for different emotional states
    segment_size = number_of_pts // 3

    # First segment: Tense, anxious (high arousal, low valence)
    segment1 = {
        'sentiment_polarity': np.linspace(0.2, 0.3, segment_size),
        'emotion_anger': np.linspace(0.7, 0.8, segment_size),
        'emotion_fear': np.linspace(0.6, 0.8, segment_size),
        'emotion_hope': np.linspace(0.2, 0.3, segment_size),
        'style_urgency': np.linspace(0.7, 0.9, segment_size),
        'hostility_confrontation': np.linspace(0.6, 0.8, segment_size),
        'peace_appeal': np.linspace(0.2, 0.1, segment_size),
    }

    # Second segment: Transition (mixed emotions)
    segment2 = {
        'sentiment_polarity': np.linspace(0.3, 0.6, segment_size),
        'emotion_anger': np.linspace(0.8, 0.4, segment_size),
        'emotion_fear': np.linspace(0.8, 0.4, segment_size),
        'emotion_hope': np.linspace(0.3, 0.6, segment_size),
        'style_urgency': np.linspace(0.9, 0.5, segment_size),
        'hostility_confrontation': np.linspace(0.8, 0.4, segment_size),
        'peace_appeal': np.linspace(0.1, 0.5, segment_size),
    }

    # Third segment: Resolution (high valence, lower arousal)
    segment3 = {
        'sentiment_polarity': np.linspace(0.6, 0.9, segment_size),
        'emotion_anger': np.linspace(0.4, 0.2, segment_size),
        'emotion_fear': np.linspace(0.4, 0.1, segment_size),
        'emotion_hope': np.linspace(0.6, 0.9, segment_size),
        'style_urgency': np.linspace(0.5, 0.3, segment_size),
        'hostility_confrontation': np.linspace(0.4, 0.1, segment_size),
        'peace_appeal': np.linspace(0.5, 0.9, segment_size),
    }

    # Combine segments
    combined_data = {}
    for key in segment1.keys():
        combined_data[key] = np.concatenate(
            [segment1[key], segment2[key], segment3[key]]
        )

    # Add some random variation
    for key in combined_data:
        noise = np.random.normal(0, 0.05, len(combined_data[key]))
        combined_data[key] = np.clip(combined_data[key] + noise, 0, 1)

    return pd.DataFrame(combined_data)


def example_with_enhanced_variation(
    df: pd.DataFrame | None = None, output_dir: str = "example_enhanced"
):
    """
    Generate music with enhanced melodic and harmonic variation.

    Args:
        df: DataFrame with emotion parameters (created if None)
        output_dir: Directory to save output files
    """
    if df is None:
        df = create_dynamic_emotion_dataframe()

    return generate_music_from_emotions(
        df,
        output_dir=output_dir,
        output_filename="enhanced_variation.mid",
        phrase_duration=3.0,  # Shorter phrases for more variation
    )


def example_with_custom_scale_type(
    df: pd.DataFrame | None = None,
    output_dir: str = "example_custom_scale_type",
    scale_type: str = "lydian",
    scale_root: int = 5,  # F
):
    """
    Generate music with a custom scale type.

    Args:
        df: DataFrame with emotion parameters (created if None)
        output_dir: Directory to save output files
        scale_type: Scale type from EXTENDED_SCALES
        scale_root: Root note of the scale (0-11, where 0=C, 1=C#, etc.)
    """
    if df is None:
        df = create_sample_dataframe()

    # Get the scale degrees for the specified type
    if scale_type in EXTENDED_SCALES:
        scale = EXTENDED_SCALES[scale_type]
    else:
        print(f"Scale type '{scale_type}' not found, using major scale")
        scale = EXTENDED_SCALES["major"]

    return generate_music_from_emotions(
        df,
        output_dir=output_dir,
        scale=scale,
        scale_root=scale_root,
        output_filename=f"{scale_type}_{get_note_name(scale_root)}_music.mid",
    )


def example_with_dramatic_emotion_changes(output_dir: str = "example_dramatic"):
    """
    Generate music with dramatic emotional changes to showcase the full range
    of musical variation.

    Args:
        output_dir: Directory to save output files
    """
    # Create a dataframe with dramatic emotional shifts
    df = create_dynamic_emotion_dataframe(30)

    return generate_music_from_emotions(
        df,
        output_dir=output_dir,
        output_filename="dramatic_emotions.mid",
        phrase_duration=4.0,
        octave_range=(3, 6),  # Wider octave range for more drama
    )


def example_with_custom_feature_mapping(
    df: pd.DataFrame | None = None, output_dir: str = "example_custom_features"
):
    """
    Generate music with a custom feature mapping to showcase how different
    emotion parameters affect the music.

    Args:
        df: DataFrame with emotion parameters (created if None)
        output_dir: Directory to save output files
    """
    if df is None:
        df = create_dynamic_emotion_dataframe()

    # Custom mapping of dataframe columns to musical dimensions
    custom_mapping = {
        'valence': ['sentiment_polarity', 'emotion_hope'],
        'arousal': ['emotion_anger', 'style_urgency'],
        'tension': ['hostility_confrontation', 'emotion_fear'],
        'complexity': ['peace_appeal'],  # Map peace to complexity
    }

    return generate_music_from_emotions(
        df,
        output_dir=output_dir,
        output_filename="custom_features.mid",
        feature_mapping=custom_mapping,
    )


# Add example function for constant scale
def example_with_constant_scale(
    df: pd.DataFrame | None = None,
    output_dir: str = "example_constant_scale",
    scale_type: str = "major",
    scale_root: int = 0,  # C
):
    """
    Generate music with a constant scale regardless of emotional changes.

    Args:
        df: DataFrame with emotion parameters (created if None)
        output_dir: Directory to save output files
        scale_type: Scale type from EXTENDED_SCALES
        scale_root: Root note of the scale (0-11, where 0=C, 1=C#, etc.)
    """
    if df is None:
        # Create a dataframe with dramatic emotional shifts
        df = create_dynamic_emotion_dataframe(30)

    # Get the scale degrees for the specified type
    if scale_type in EXTENDED_SCALES:
        scale = EXTENDED_SCALES[scale_type]
    else:
        print(f"Scale type '{scale_type}' not found, using major scale")
        scale = EXTENDED_SCALES["major"]

    root_name = get_note_name(scale_root)

    return generate_music_from_emotions(
        df,
        output_dir=output_dir,
        scale=scale,
        scale_root=scale_root,
        force_constant_scale=True,  # This is the key parameter
        use_chord_aware_melody=True,
        output_filename=f"constant_{scale_type}_{root_name}_scale.mid",
    )


# Example demonstrating chord-aware melodies
def example_with_chord_aware_melody(
    df: pd.DataFrame | None = None,
    output_dir: str = "example_chord_aware",
    scale_type: str = "major",
    scale_root: int = 0,  # C
):
    """
    Generate music with chord-aware melody generation.

    Args:
        df: DataFrame with emotion parameters (created if None)
        output_dir: Directory to save output files
        scale_type: Scale type from EXTENDED_SCALES
        scale_root: Root note of the scale (0-11, where 0=C, 1=C#, etc.)
    """
    if df is None:
        # Create a dataframe with dramatic emotional shifts
        df = create_dynamic_emotion_dataframe(30)

    # Get the scale degrees for the specified type
    if scale_type in EXTENDED_SCALES:
        scale = EXTENDED_SCALES[scale_type]
    else:
        print(f"Scale type '{scale_type}' not found, using major scale")
        scale = EXTENDED_SCALES["major"]

    root_name = get_note_name(scale_root)

    return generate_music_from_emotions(
        df,
        output_dir=output_dir,
        scale=scale,
        scale_root=scale_root,
        force_constant_scale=True,
        use_chord_aware_melody=True,  # Ensure chord-aware melody generation
        output_filename=f"chord_aware_{scale_type}_{root_name}.mid",
    )


def create_simple_emotion_dataframe(number_of_pts: int = 16) -> pd.DataFrame:
    """
    Create a sample dataframe with gentle, simple emotional changes.

    Args:
        number_of_pts: Number of data points to generate

    Returns:
        DataFrame with synthetic emotion parameters suitable for simple music
    """
    # Create a simple emotional arc that moves from neutral to slightly positive
    # with low complexity and tension

    # Start with neutral valence, gradually increasing
    valence = np.linspace(0.5, 0.7, number_of_pts)

    # Keep arousal moderate and steady
    arousal = np.ones(number_of_pts) * 0.4

    # Low tension throughout
    tension = np.ones(number_of_pts) * 0.2

    # Very low complexity
    complexity = np.ones(number_of_pts) * 0.2

    # Create the dataframe with our controlled parameters
    df = pd.DataFrame(
        {
            'valence': valence,
            'arousal': arousal,
            'tension': tension,
            'complexity': complexity,
        }
    )

    return df


def example_simple_music(
    output_dir: str = "example_simple_music",
    scale_type: str = "major",
    scale_root: int = 0,  # C
):
    """
    Generate very simple, minimalist music with a constant scale and few notes.

    Args:
        output_dir: Directory to save output files
        scale_type: Scale type from EXTENDED_SCALES
        scale_root: Root note of the scale (0-11, where 0=C, 1=C#, etc.)
    """
    # Create a simple dataframe with controlled parameters
    df = create_simple_emotion_dataframe(16)

    # Get the scale degrees for the specified type
    if scale_type in EXTENDED_SCALES:
        scale = EXTENDED_SCALES[scale_type]
    else:
        print(f"Scale type '{scale_type}' not found, using major scale")
        scale = EXTENDED_SCALES["major"]

    root_name = get_note_name(scale_root)

    # Define simple custom chord progressions with just major and minor chords
    simple_chord_mappings = {
        'complex': [[('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')]],
        'tense': [[('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')]],
        'positive': [[('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')]],
        'negative': [[('C', 'minor'), ('F', 'minor'), ('G', 'minor'), ('C', 'minor')]],
        'dreamy': [[('C', 'major'), ('A', 'minor'), ('F', 'major'), ('G', 'major')]],
        'triumphant': [
            [('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')]
        ],
        'mysterious': [
            [('A', 'minor'), ('F', 'major'), ('G', 'major'), ('A', 'minor')]
        ],
    }

    # Define simpler duration patterns with fewer variations
    simple_durations = {
        'high_arousal': [[0.5, 0.5, 0.5, 0.5]],  # Simple quarter notes
        'medium_arousal': [[1.0, 1.0]],  # Simple half notes
        'low_arousal': [[2.0, 2.0]],  # Simple whole notes
    }

    return generate_music_from_emotions(
        df,
        output_dir=output_dir,
        scale=scale,
        scale_root=scale_root,
        force_constant_scale=True,
        use_chord_aware_melody=True,
        chord_mappings=simple_chord_mappings,
        duration_mappings=simple_durations,
        phrase_duration=4.0,  # Longer phrases for fewer chord changes
        octave_range=(4, 5),  # Narrower octave range
        tempo_range=(60, 80),  # Slower, more consistent tempo
        output_filename=f"simple_{scale_type}_{root_name}.mid",
    )


def example_simple_pentatonic(
    output_dir: str = "example_simple_pentatonic", scale_root: int = 0  # C
):
    """
    Generate extremely simple music using pentatonic scale for a more
    consonant, straightforward sound.

    Args:
        output_dir: Directory to save output files
        scale_root: Root note of the scale (0-11, where 0=C, 1=C#, etc.)
    """
    # Create a simple dataframe with controlled parameters
    # Even fewer points for less variation
    df = create_simple_emotion_dataframe(8)

    # Use pentatonic scale for maximum simplicity and consonance
    scale = EXTENDED_SCALES["pentatonic_major"]

    root_name = get_note_name(scale_root)

    # Define very simple chord progressions with just I-IV-V-I
    simple_chord_mappings = {
        'complex': [[('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')]],
        'tense': [[('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')]],
        'positive': [[('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')]],
        'negative': [[('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')]],
        'dreamy': [[('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')]],
        'triumphant': [
            [('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')]
        ],
        'mysterious': [
            [('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')]
        ],
    }

    # Define even simpler duration patterns - mostly half notes
    simple_durations = {
        'high_arousal': [[1.0, 1.0, 1.0, 1.0]],  # Simple half notes
        'medium_arousal': [[1.0, 1.0, 1.0, 1.0]],  # Simple half notes
        'low_arousal': [[2.0, 2.0]],  # Simple whole notes
    }

    return generate_music_from_emotions(
        df,
        output_dir=output_dir,
        scale=scale,
        scale_root=scale_root,
        force_constant_scale=True,
        use_chord_aware_melody=True,
        chord_mappings=simple_chord_mappings,
        duration_mappings=simple_durations,
        phrase_duration=8.0,  # Much longer phrases for fewer changes
        octave_range=(4, 4),  # Single octave for extreme simplicity
        tempo_range=(60, 60),  # Constant tempo
        output_filename=f"simple_pentatonic_{root_name}.mid",
    )


# Entry point for running examples
if __name__ == "__main__":
    print("Emotion Music Generator")
    print("======================")
    print("1. Running enhanced variation example...")
    example_with_enhanced_variation()

    print("\n2. Running constant scale example (C major)...")
    example_with_constant_scale(scale_type="major", scale_root=0)

    print("\n3. Running chord-aware melody example...")
    example_with_chord_aware_melody()

    print("\nAll examples completed. Check output directories for generated files.")
