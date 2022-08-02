from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from typing import Optional


def parse_ort_file(ort_path: Path) -> pd.DataFrame:
    """Parse the ort (orthographic description) file

    each row = a single time-range transcript
    """
    with open(ort_path, "rb") as f:
        lines = f.read().decode("iso-8859-1").split("\n")

    # The first three lines are always the same.
    #  1. 	File type = "ooTextFile short"
    #  2. 	"TextGrid"
    #  3. 	{empty line}
    assert lines[0] == 'File type = "ooTextFile short"'
    assert lines[1] == '"TextGrid"'
    assert lines[2] == ""

    # On lines 4 and 5 a description is given of the timespan involved.
    # Time here is expressed in terms of the number of seconds, using three decimals.
    # t_rec_start, t_rec_end = float(lines[3]), float(lines[4])

    # Lines 6 and 7 describe the number of tiers that occur in the file.
    assert lines[5] == "<exists>"
    # nb_tiers = int(lines[6])

    # Lines 8 up to and including 12 contain information about the first tier.
    assert lines[7] == '"IntervalTier"', f"Expected 'IntervalTier', got '{lines[7]}'"

    assert lines[-1] == "", "The last line should be empty."
    lines = lines[:-1]

    # exclude the interviewer user from component B
    i = 7
    df_list = []
    speaker_name = ""
    while i < len(lines):
        if lines[i] == '"IntervalTier"':
            speaker_name = lines[i + 1]
            # t_start, t_stop = list(map(float, lines[i + 2: i + 4]))
            # nb_intervals = int(lines[i + 4])
            i += 5
        else:
            t_start, t_stop = lines[i: i + 2]
            transcript = lines[i + 2]
            df_list.append(
                {
                    "speaker_name": speaker_name.strip('"'),
                    "t_start": t_start,
                    "t_stop": t_stop,
                    "transcript": transcript.strip('"'),
                }
            )

            i += 3

    df_ort = pd.DataFrame(df_list)
    df_ort[["t_start", "t_stop"]] = df_ort[["t_start", "t_stop"]].apply(pd.to_numeric)
    df_ort = df_ort.sort_values(by=["speaker_name", "t_start"]).reset_index(drop=True)
    df_ort["rec_name"] = ort_path.name.split(".")[0]
    return df_ort


def parse_ort_file_agg(ort_path: Path) -> pd.DataFrame:
    """Parses an ORT file and aggregates consecutive transcriptions by the same speaker.

    Parameters
    ----------
    ort_path : Path
        The path of the ort file to parse.

    Returns
    -------
    pd.DataFrame
        The parsed ORT file in a dataframe format.
    """
    with open(ort_path, "rb") as f:
        lines = f.read().decode("iso-8859-1").split("\n")

    # The first three lines are always the same.
    #  1. 	File type = "ooTextFile short"
    #  2. 	"TextGrid"
    #  3. 	{empty line}
    assert lines[0] == 'File type = "ooTextFile short"'
    assert lines[1] == '"TextGrid"'
    assert lines[2] == ""

    # On lines 4 and 5 a description is given of the timespan involved.
    # Time here is expressed in terms of the number of seconds, using three decimals.
    # t_rec_start, t_rec_end = float(lines[3]), float(lines[4])

    # Lines 6 and 7 describe the number of tiers that occur in the file.
    assert lines[5] == "<exists>"
    # nb_tiers = int(lines[6])

    # Lines 8 up to and including 12 contain information about the first tier.
    assert lines[7] == '"IntervalTier"', f"Expected 'IntervalTier', got '{lines[7]}'"

    assert lines[-1] == "", "The last line should be empty."
    lines = lines[:-1]

    # exclude the interviewer user from component B
    i = 7
    df_list = []
    speaker_name = ""

    @dataclass
    class Block:  # data container
        start: object
        stop: object
        transcript: Optional[str]

    block = Block(None, None, None)

    def add_clear_block(df_l: list, b: Block):
        if b.transcript is None:
            return  # no block to add

        df_l.append(
            {
                "speaker_name": speaker_name,
                "t_start": b.start,
                "t_stop": b.stop,
                "transcript": b.transcript,
            }
        )
        b.transcript, b.start, b.stop = None, None, None

    while i < len(lines):
        # A new interval starts
        if lines[i] == '"IntervalTier"':
            # Base case -> add existing block to df_list
            if block.transcript is not None:
                add_clear_block(df_list, block)

            # Get the speaker name from the new interval
            speaker_name = lines[i + 1].strip('"')

            # the other fields aren't that interesting
            # t_start, t_stop = list(map(float, lines[i + 2: i + 4]))
            # nb_intervals = int(lines[i + 4])
            i += 5
        else:
            t_start, t_stop = lines[i: i + 2]
            transcript = lines[i + 2].strip('"')

            if len(transcript) == 0:
                # empty transcript (other person speaking) -> add to block & clear
                add_clear_block(df_list, block)
            elif block.transcript is None:
                # Base case - empty block -> start new block
                block.start = t_start
                block.stop = t_stop
                block.transcript = transcript
            elif block.transcript is not None:
                if block.stop == t_start:  # we are able to extend the block
                    # print('extending block')
                    block.transcript = block.transcript + " " + transcript
                    block.stop = t_stop
                else:  # We cannot extend the block
                    # add the previous block and start a new one
                    add_clear_block(df_list, block)

                    block.start, block.stop = t_start, t_stop
                    block.transcript = transcript

            i += 3

    df_ort = pd.DataFrame(df_list)
    df_ort[["t_start", "t_stop"]] = df_ort[["t_start", "t_stop"]].apply(pd.to_numeric)
    df_ort["transcr_len"] = df_ort["transcript"].apply(len)
    df_ort["duration_s"] = df_ort["t_stop"] - df_ort["t_start"]
    df_ort["rec_name"] = ort_path.name.split(".")[0]
    return df_ort
