"""Various util functions for the Corpus Gesproken Nederlands (CGN) dataset"""
import numpy as np
import pandas as pd
import torchaudio

from ..path_conf import cgn_root_dir


def listen_to_audio(
    metadata_row: pd.Series, margin_s: float = 1, display_file_path: bool = True
):
    """Listen to an audio file in the CGN dataset"""
    from IPython.display import Audio

    recording_name = None
    for c in ["rec_name", "recordingID"]:
        if c in metadata_row:
            recording_name = metadata_row[c]
            break
    assert recording_name is not None

    # Load the audio data
    file_path = list(cgn_root_dir.glob(f"**/{recording_name}.wav"))
    assert len(file_path) == 1

    wav, sr = torchaudio.load(file_path[0], normalize=True)
    wav = wav.numpy().ravel()

    if display_file_path:
        print(file_path[0], sr, wav.shape)

    start_idx, end_idx = int(max(0, sr * metadata_row.t_start - margin_s * sr)), int(
        sr * metadata_row.t_stop + margin_s * sr
    )
    # print(wav)
    # print("start_idx:", start_idx, "end_idx:", end_idx)
    slice = wav[start_idx:end_idx]
    print(
        "start:",
        round(metadata_row.t_start, 2),
        "stop:",
        round(metadata_row.t_stop, 2),
        "margin:",
        margin_s,
    )

    return Audio(slice, rate=sr, autoplay=False)


def get_audio(metadata_row: pd.Series) -> np.ndarray:
    recording_name = None
    for c in ["rec_name", "recordingID"]:
        if c in metadata_row:
            recording_name = metadata_row[c]
            break
    assert recording_name is not None

    # Load the audio data
    file_path = list(cgn_root_dir.glob(f"**/{recording_name}.wav"))
    assert len(file_path) == 1

    wav, sr = torchaudio.load(file_path[0], normalize=True)
    assert sr == 16_000
    wav = wav.numpy().ravel()
    return wav
