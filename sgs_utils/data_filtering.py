import pandas as pd


def _get_silent_recording_mask(df):
    # no recording audio
    bad_recorgins = (
        (
            # Verified manually by listening
            (df["ID"] == "2e98e07d-5d7c-469b-aee1-2421856671b0")
            & (df["pic_name"] == "Picture 110")
        )
        | (
            # Verified manually by listening
            (df["ID"] == "2e98e07d-5d7c-469b-aee1-2421856671b0")
            & (df["pic_name"] == "Rafd090_49_Caucasian_male_neutral_frontal")
        )
        | (
            # Verified manually by listening
            (df["ID"] == "2e98e07d-5d7c-469b-aee1-2421856671b0")
            & (df["pic_name"] == "Rafd090_05_Caucasian_male_neutral_frontal")
        )
        | (
            # Verified manually by listening
            (df["ID"] == "1d259634-7d1a-461c-9098-5be8e94b105b")
            & (df["pic_name"] == "Rafd090_36_Caucasian_male_neutral_frontal")
        )
        | (
            # Verified manually by listening
            (df["ID"] == "63defe6b-db35-4b9c-b53d-90df6612beaa")
            & (df["pic_name"] == "Rafd090_36_Caucasian_male_neutral_frontal")
        )
        | (
            # No recording audio in the last 15 seconds
            (df["ID"] == "8f42a931-7bd1-4536-bec3-a5f24da3c61f")
            & (df["pic_name"] == "Rafd090_36_Caucasian_male_neutral_frontal")
        )
    )
    return bad_recorgins


def get_valid_audio_mask(df: pd.DataFrame) -> pd.Series:
    ids_to_skip = [
        # This user has 14 PiSCES and 10 Radboud recordings under 15 seconds
        "a2dbee10-fc31-42ec-89f8-9b4e5fa74b7f",  # Too much short sessions

        # A lot of background noise - openSMILE can't properly deal with this data
        "28d01050-e4e0-4115-a5e9-09c8cb917fb0",

        # Low quality user data
        "0c28e160-d7bb-4b17-829c-c9ebc7ad0a1f",
        "3a8266c9-7087-45fe-98e4-6c9c001c0050",
        "42c842a5-7051-44c3-af42-cf824fded959",
        "d7fbc4b9-5f94-4639-b82b-6b1592b2ad3a",
    ]
    bad_id_mask = df["ID"].isin(ids_to_skip)
    no_audio_mask = _get_silent_recording_mask(df)

    return ~bad_id_mask & ~no_audio_mask
