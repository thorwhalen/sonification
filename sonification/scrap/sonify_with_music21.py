"""
Emotion Music Generator

A customizable music generation tool that creates MIDI music based on emotion parameters.
This module uses music21 to generate melodies and accompaniments that reflect emotional
timeseries data.

Key Components:
- Feature Mapping: Maps raw emotion features to musical dimensions (valence, arousal, etc.)
- Scale Selection: Customizable scales for different emotional states
- Chord Progression: Customizable chord progressions based on emotional states
- Melodic Generation: Creates melodies reflecting emotional parameters
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
from typing import List, Dict, Optional, Union, Tuple, Callable, Any
import time

# Default module constants - can be modified if needed
DEFAULT_SCALE = [0, 2, 4, 5, 7, 9, 11]  # C Major (0=C, 1=C#, etc.)
DEFAULT_SCALE_ROOT = 0  # C
DEFAULT_TEMPO_RANGE = (60, 160)  # BPM range mapped to arousal
DEFAULT_PHRASE_DURATION = 4.0  # Beats per emotion data point
DEFAULT_OCTAVE_RANGE = (3, 5)  # Base octave range for melody
DEFAULT_VELOCITY_RANGE = (70, 110)  # MIDI velocity range for dynamics
DEFAULT_DURATION_MAPPINGS = {
    'high_arousal': [0.25, 0.25, 0.5, 0.25, 0.25, 0.5],  # Fast notes
    'medium_arousal': [0.5, 0.5, 0.5, 0.5],             # Medium notes
    'low_arousal': [1.0, 0.5, 1.0, 1.5]                 # Slow notes
}

# Default feature groupings - can be customized by user
DEFAULT_FEATURE_MAPPING = {
    'valence': ['sentiment_polarity', 'emotion_hope', 'peace_appeal'],
    'arousal': ['emotion_anger', 'emotion_fear', 'style_urgency', 'moral_outrage'],
    'tension': ['hostility_confrontation', 'military_intensity', 'intent_provocation'],
    'complexity': ['assertion_strength', 'factual_speculative', 'intent_persuasion']
}

# Chord mappings - can be modified if needed
DEFAULT_CHORD_MAPPINGS = {
    'complex': [
        ('C', 'major-seventh'),
        ('A', 'minor-seventh'),
        ('F', 'major-seventh'),
        ('D', 'minor-seventh')
    ],
    'tense': [
        ('C', 'minor'),
        ('G', 'dominant-seventh'),
        ('A', 'diminished'),
        ('D', 'half-diminished-seventh')
    ],
    'positive': [
        ('C', 'major'),
        ('G', 'major'),
        ('A', 'minor'),
        ('F', 'major')
    ],
    'negative': [
        ('C', 'minor'),
        ('G', 'minor'),
        ('E-flat', 'major'),
        ('F', 'minor')
    ]
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
            normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
    
    return normalized_df


def map_features(
    df: pd.DataFrame, 
    feature_mapping: Dict[str, List[str]] = None
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


def transpose_scale(
    base_scale: List[int], 
    root: int = 0
) -> List[int]:
    """
    Transpose a scale to a new root note.
    
    Args:
        base_scale: List of scale degrees (0-11)
        root: New root note (0-11, where 0=C, 1=C#, etc.)
        
    Returns:
        Transposed scale
    """
    return [(note + root) % 12 for note in base_scale]


def transpose_chord_progression(
    chord_progression: List[Tuple[str, str]],
    semitones: int
) -> List[Tuple[str, str]]:
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
    # Alternative flat names (to convert flat notation to our sharp-based list)
    flat_to_sharp = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#', 
                      'Df': 'C#', 'Ef': 'D#', 'Gf': 'F#', 'Af': 'G#', 'Bf': 'A#'}
    
    transposed_progression = []
    for root, quality in chord_progression:
        # Convert 'E-flat' or 'E-' notation to 'Eb' format
        if '-flat' in root:
            root = root.replace('-flat', 'b')
        if '-' in root and len(root) > 1 and root[1] == '-':
            root = root[0] + 'b' + root[2:]
        
        # Handle flats properly
        if len(root) > 1 and root[1] == 'b':
            # Convert flat notation to the equivalent sharp
            if root in flat_to_sharp:
                base_idx = note_names.index(flat_to_sharp[root])
            else:
                # Handle multi-character flat notation
                base_note = root[0]
                base_idx = note_names.index(base_note)
                base_idx = (base_idx - 1) % 12
        else:
            # Get the base note without any modifiers
            base_note = root[0].upper()
            
            # Find the index of the base note
            base_idx = note_names.index(base_note)
            
            # Handle sharps in the root
            if len(root) > 1 and root[1] == '#':
                base_idx = (base_idx + 1) % 12
        
        # Calculate new root index
        new_idx = (base_idx + semitones) % 12
        
        # Get new root name
        new_root = note_names[new_idx]
        
        # Add to the transposed progression
        transposed_progression.append((new_root, quality))
    
    return transposed_progression


def get_chord_notes(
    root: str, 
    quality: str
) -> List[str]:
    """
    Get the notes for a chord based on root and quality.
    
    Args:
        root: Root note name (e.g., 'C', 'F#')
        quality: Chord quality (e.g., 'major', 'minor', 'major-seventh')
        
    Returns:
        List of note names in the chord
    """
    # Handle flat notation in root
    if '-flat' in root:
        root = root.replace('-flat', 'b')
    if '-' in root and len(root) > 1 and root[1] == '-':
        root = root[0] + 'b' + root[2:]
    
    # Create a music21 chord based on the quality
    if quality == 'major':
        chord = [root + '3', root + '4', 
                music21.note.Note(root + '3').transpose('M3').nameWithOctave,
                music21.note.Note(root + '3').transpose('P5').nameWithOctave]
    elif quality == 'minor':
        chord = [root + '3', root + '4',
                music21.note.Note(root + '3').transpose('m3').nameWithOctave,
                music21.note.Note(root + '3').transpose('P5').nameWithOctave]
    elif quality == 'major-seventh':
        chord = [root + '3',
                music21.note.Note(root + '3').transpose('M3').nameWithOctave,
                music21.note.Note(root + '3').transpose('P5').nameWithOctave,
                music21.note.Note(root + '3').transpose('M7').nameWithOctave]
    elif quality == 'minor-seventh':
        chord = [root + '3',
                music21.note.Note(root + '3').transpose('m3').nameWithOctave,
                music21.note.Note(root + '3').transpose('P5').nameWithOctave,
                music21.note.Note(root + '3').transpose('m7').nameWithOctave]
    elif quality == 'dominant-seventh':
        chord = [root + '3',
                music21.note.Note(root + '3').transpose('M3').nameWithOctave,
                music21.note.Note(root + '3').transpose('P5').nameWithOctave,
                music21.note.Note(root + '3').transpose('m7').nameWithOctave]
    elif quality == 'diminished':
        chord = [root + '3',
                music21.note.Note(root + '3').transpose('m3').nameWithOctave,
                music21.note.Note(root + '3').transpose('d5').nameWithOctave]
    elif quality == 'half-diminished-seventh':
        chord = [root + '3',
                music21.note.Note(root + '3').transpose('m3').nameWithOctave,
                music21.note.Note(root + '3').transpose('d5').nameWithOctave,
                music21.note.Note(root + '3').transpose('m7').nameWithOctave]
    else:
        # Default to major triad if quality not recognized
        chord = [root + '3', root + '4',
                music21.note.Note(root + '3').transpose('M3').nameWithOctave,
                music21.note.Note(root + '3').transpose('P5').nameWithOctave]
    
    return chord


def select_chord_progression(
    valence: float,
    tension: float,
    complexity: float,
    chord_mappings: Dict[str, List[Tuple[str, str]]] = None,
    scale_root: int = 0
) -> List[Tuple[str, str]]:
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
        chord_mappings = DEFAULT_CHORD_MAPPINGS
    
    # Select the base chord progression based on emotional state
    if complexity > 0.7:
        progression = chord_mappings['complex']
    elif tension > 0.7:
        progression = chord_mappings['tense']
    elif valence > 0.6:
        progression = chord_mappings['positive']
    else:
        progression = chord_mappings['negative']
    
    # Transpose the progression if needed
    if scale_root != 0:
        progression = transpose_chord_progression(progression, scale_root)
    
    return progression


def create_emotion_music(
    df: pd.DataFrame,
    output_file: str = "emotion_music.mid",
    scale: List[int] = None,
    scale_root: int = DEFAULT_SCALE_ROOT,
    feature_mapping: Dict[str, List[str]] = None,
    chord_mappings: Dict[str, List[Tuple[str, str]]] = None,
    phrase_duration: float = DEFAULT_PHRASE_DURATION,
    tempo_range: Tuple[int, int] = DEFAULT_TEMPO_RANGE,
    octave_range: Tuple[int, int] = DEFAULT_OCTAVE_RANGE,
    velocity_range: Tuple[int, int] = DEFAULT_VELOCITY_RANGE,
    duration_mappings: Dict[str, List[float]] = None,
    normalize: bool = True,
    progress_callback: Callable[[int, int], None] = None
) -> str:
    """
    Create MIDI music based on emotion data.
    
    Args:
        df: DataFrame with emotion parameters
        output_file: Path for output MIDI file
        scale: Scale to use for the piece (list of integers 0-11)
        scale_root: Root note of the scale (0-11, where 0=C, 1=C#, etc.)
        feature_mapping: Dictionary mapping musical dimensions to lists of feature names
        chord_mappings: Dictionary mapping emotional states to chord progressions
        phrase_duration: Duration of each emotion segment in beats
        tempo_range: Range of tempos to map arousal to (min, max)
        octave_range: Range of octaves for melody (min, max)
        velocity_range: Range of MIDI velocities for dynamics (min, max)
        duration_mappings: Dictionary mapping arousal levels to note durations
        normalize: Whether to normalize emotion values to 0-1 range
        progress_callback: Function to call with progress updates (current, total)
        
    Returns:
        Path to the generated MIDI file
    """
    
    # Set defaults
    if scale is None:
        scale = DEFAULT_SCALE
    if feature_mapping is None:
        feature_mapping = DEFAULT_FEATURE_MAPPING
    if chord_mappings is None:
        chord_mappings = DEFAULT_CHORD_MAPPINGS
    if duration_mappings is None:
        duration_mappings = DEFAULT_DURATION_MAPPINGS
    
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
    
    # Current position in the score
    current_offset = 0.0
    
    # Create music21 scales based on the provided scale degrees
    scale_degrees = [scale_root + note for note in scale]
    
    # Create major and minor versions for different emotional contexts
    major_scale = music21.scale.ConcreteScale(
        pitches=[music21.pitch.Pitch(n % 12) for n in scale_degrees]
    )
    minor_scale = music21.scale.ConcreteScale(
        pitches=[music21.pitch.Pitch((scale_root + n) % 12) for n in [0, 2, 3, 5, 7, 8, 10]]
    )
    
    # Process each row in the dataframe
    total_rows = len(mapped_df)
    for idx, row in mapped_df.iterrows():
        # Update progress if callback provided
        if progress_callback and idx % max(1, total_rows // 10) == 0:
            progress_callback(idx, total_rows)
        
        # Extract emotional dimensions
        valence = row.get('valence', 0.5)  # positive/negative sentiment
        arousal = row.get('arousal', 0.5)  # energy/intensity
        tension = row.get('tension', 0.5)  # conflict/dissonance
        complexity = row.get('complexity', 0.5)  # musical complexity
        
        # 1. Select scale based on emotional state
        if valence > 0.7:
            current_scale = major_scale  # More positive emotion
        elif valence < 0.4:
            current_scale = minor_scale  # More negative emotion
        else:
            current_scale = major_scale if valence > 0.5 else minor_scale
        
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
            # Get chord notes
            chord_notes = get_chord_notes(root, quality)
            
            # Create chord
            chord = music21.chord.Chord(chord_notes)
            
            # Set chord duration
            chord.duration = music21.duration.Duration(phrase_duration / len(chord_progression))
            
            # Set chord loudness based on arousal
            velocity_min, velocity_max = velocity_range
            chord_velocity = int(velocity_min + arousal * (velocity_max - velocity_min))
            chord.volume.velocity = min(127, chord_velocity)
            
            # Position chord in the accompaniment part
            chord_offset = phrase_offset + (i * phrase_duration / len(chord_progression))
            accomp_part.insert(chord_offset, chord)
        
        # 5. Create melody
        num_notes = 8  # Number of notes per phrase
        
        # Select appropriate note durations based on arousal
        if arousal > 0.7:
            durations = duration_mappings['high_arousal']
        elif arousal < 0.4:
            durations = duration_mappings['low_arousal']
        else:
            durations = duration_mappings['medium_arousal']
        
        for i in range(num_notes):
            # Scale degree selection based on emotional state
            if valence > 0.7 and tension < 0.3:
                # Happy, stable melody - use stable scale degrees
                scale_degrees = [1, 3, 5, 8]
                scale_degree = scale_degrees[i % len(scale_degrees)]
            elif tension > 0.7:
                # Tense melody - use more unstable scale degrees
                scale_degrees = [2, 4, 6, 7]
                scale_degree = scale_degrees[i % len(scale_degrees)]
            else:
                # Mixed melody
                scale_degree = ((i * 2) % 7) + 1
            
            # Get pitch from scale
            pitch = current_scale.pitchFromDegree(scale_degree)
            
            # Adjust octave based on phrase position and arousal
            octave_min, octave_max = octave_range
            base_octave = octave_min
            octave_adj = 1 if i % 3 == 0 else 0
            octave_adj += 1 if arousal > 0.7 and i % 4 == 2 else 0
            pitch.octave = min(octave_max, base_octave + octave_adj)
            
            # Get note duration
            duration = durations[i % len(durations)]
            
            # Create note
            note = music21.note.Note(pitch)
            note.duration = music21.duration.Duration(duration)
            
            # Set velocity based on arousal and position
            velocity_min, velocity_max = velocity_range
            note_velocity = int(velocity_min + arousal * (velocity_max - velocity_min))
            note_velocity += 10 if i % 4 == 0 else 0  # Emphasize downbeats
            note.volume.velocity = min(127, note_velocity)
            
            # Position note in the melody part
            note_offset = phrase_offset + (i * phrase_duration / num_notes)
            melody_part.insert(note_offset, note)
        
        # Move to next phrase
        current_offset += phrase_duration
    
    # Final progress update
    if progress_callback:
        progress_callback(total_rows, total_rows)
    
    # Add parts to the score
    score.insert(0, melody_part)
    score.insert(0, accomp_part)
    
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
    feature_mapping: Dict[str, List[str]] = None,
    show_plot: bool = False,
    title: str = 'Emotion Parameters Over Time',
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300
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
            'sentiment_polarity', 'emotion_anger', 'emotion_fear',
            'emotion_hope', 'hostility_confrontation', 'peace_appeal'
        ]
        key_features = [col for col in potential_features if col in df.columns][:6]  # Limit to 6
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


def generate_music_from_emotions(
    df: pd.DataFrame,
    output_dir: str = "emotion_music",
    output_filename: str = "emotion_music.mid",
    scale: List[int] = None,
    scale_root: int = DEFAULT_SCALE_ROOT,
    feature_mapping: Dict[str, List[str]] = None,
    chord_mappings: Dict[str, List[Tuple[str, str]]] = None,
    phrase_duration: float = DEFAULT_PHRASE_DURATION,
    tempo_range: Tuple[int, int] = DEFAULT_TEMPO_RANGE,
    octave_range: Tuple[int, int] = DEFAULT_OCTAVE_RANGE,
    velocity_range: Tuple[int, int] = DEFAULT_VELOCITY_RANGE,
    duration_mappings: Dict[str, List[float]] = None,
    visualize: bool = True,
    normalize: bool = True
) -> Dict[str, str]:
    """
    Generate music and visualization from emotion data.
    
    Args:
        df: DataFrame with emotion parameters
        output_dir: Directory to save output files
        output_filename: Filename for the MIDI output
        scale: Scale to use for the piece (list of integers 0-11)
        scale_root: Root note of the scale (0-11, where 0=C, 1=C#, etc.)
        feature_mapping: Dictionary mapping musical dimensions to lists of feature names
        chord_mappings: Dictionary mapping emotional states to chord progressions
        phrase_duration: Duration of each emotion segment in beats
        tempo_range: Range of tempos to map arousal to (min, max)
        octave_range: Range of octaves for melody (min, max)
        velocity_range: Range of MIDI velocities for dynamics (min, max)
        duration_mappings: Dictionary mapping arousal levels to note durations
        visualize: Whether to create visualization
        normalize: Whether to normalize emotion values to 0-1 range
        
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
        print(f"Progress: {current}/{total} data points processed ({current/total*100:.1f}%)")
    
    # Generate visualization
    if visualize:
        vis_file = os.path.join(output_dir, "emotion_visualization.png")
        visualize_emotion_data(
            df, 
            vis_file, 
            feature_mapping=feature_mapping
        )
        output_files['visualization'] = vis_file
    
    # Generate music
    try:
        music_file = os.path.join(output_dir, output_filename)
        create_emotion_music(
            df,
            music_file,
            scale=scale,
            scale_root=scale_root,
            feature_mapping=feature_mapping,
            chord_mappings=chord_mappings,
            phrase_duration=phrase_duration,
            tempo_range=tempo_range,
            octave_range=octave_range,
            velocity_range=velocity_range,
            duration_mappings=duration_mappings,
            normalize=normalize,
            progress_callback=progress_callback
        )
        output_files['music'] = music_file
    except Exception as e:
        print(f"Error generating music: {e}")
    
    # Report execution time
    execution_time = time.time() - start_time
    print(f"Music generation completed in {execution_time:.2f} seconds")
    
    # Print summary
    print("\nGenerated files:")
    for file_type, file_path in output_files.items():
        print(f"  - {file_type}: {file_path}")
    
    return output_files


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
        'emotion_fear': np.cos(np.linspace(0, 2*np.pi, number_of_pts)) * 0.2 + 0.5,
        'emotion_hope': np.sin(np.linspace(0, 2*np.pi, number_of_pts)) * 0.3 + 0.6,
        'style_urgency': np.sin(np.linspace(0, np.pi, number_of_pts)) * 0.4 + 0.5,
        'hostility_confrontation': np.abs(np.sin(np.linspace(0, 3*np.pi, number_of_pts))) * 0.5 + 0.3,
        'peace_appeal': np.cos(np.linspace(0, 3*np.pi, number_of_pts)) * 0.3 + 0.6
    }
    
    return pd.DataFrame(sample_data)


def example_with_default_settings(df: Optional[pd.DataFrame] = None, output_dir: str = "example_default"):
    """
    Generate music with default settings.
    
    Args:
        df: DataFrame with emotion parameters (created if None)
        output_dir: Directory to save output files
    """
    if df is None:
        df = create_sample_dataframe()
        
    return generate_music_from_emotions(
        df, 
        output_dir=output_dir
    )


def example_with_custom_scale(
    df: Optional[pd.DataFrame] = None, 
    output_dir: str = "example_custom_scale",
    scale: List[int] = [0, 2, 3, 5, 7, 8, 10],  # Natural minor
    scale_root: int = 2  # D
):
    """
    Generate music with a custom scale.
    
    Args:
        df: DataFrame with emotion parameters (created if None)
        output_dir: Directory to save output files
        scale: Scale degrees (0-11)
        scale_root: Root note of the scale (0-11, where 0=C, 1=C#, etc.)
    """
    if df is None:
        df = create_sample_dataframe()
        
    return generate_music_from_emotions(
        df, 
        output_dir=output_dir,
        scale=scale,
        scale_root=scale_root,
        output_filename=f"scale_{scale_root}_music.mid"
    )


def example_with_custom_parameters(
    df: Optional[pd.DataFrame] = None, 
    output_dir: str = "example_custom_params"
):
    """
    Generate music with customized musical parameters.
    
    Args:
        df: DataFrame with emotion parameters (created if None)
        output_dir: Directory to save output files
    """
    if df is None:
        df = create_sample_dataframe()
        
    return generate_music_from_emotions(
        df,
        output_dir=output_dir,
        output_filename="custom_params.mid",
        phrase_duration=2.0,  # Shorter phrases
        tempo_range=(80, 140),  # Narrower tempo range
        octave_range=(4, 6),  # Higher register
        duration_mappings={
            'high_arousal': [0.25, 0.25, 0.25, 0.25],  # Very fast notes
            'medium_arousal': [0.5, 0.25, 0.25, 0.5],  # Medium notes
            'low_arousal': [1.0, 1.0, 0.5, 0.5]        # Slower notes
        }
    )


def example_with_custom_mapping(
    df: Optional[pd.DataFrame] = None, 
    output_dir: str = "example_custom_mapping"
):
    """
    Generate music with a custom feature mapping.
    
    Args:
        df: DataFrame with emotion parameters (created if None)
        output_dir: Directory to save output files
    """
    if df is None:
        df = create_sample_dataframe()
    
    # Custom mapping of dataframe columns to musical dimensions
    custom_mapping = {
        'valence': ['sentiment_polarity'],  # Only use sentiment polarity for valence
        'arousal': ['emotion_anger', 'style_urgency'],  # Only use anger and urgency
        'tension': ['hostility_confrontation'],
        'complexity': ['peace_appeal']  # Map peace to complexity
    }
    
    return generate_music_from_emotions(
        df,
        output_dir=output_dir,
        output_filename="custom_mapping.mid",
        feature_mapping=custom_mapping
    )


# You can run these from a notebook or script
if __name__ == "__main__":
    print("Import this module and run the example functions directly.")