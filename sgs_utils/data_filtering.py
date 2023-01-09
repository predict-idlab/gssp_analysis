import pandas as pd


def _silent_recording_mask(df: pd.DataFrame) -> pd.Series:
    # no recording audio
    bad_recordings = (
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
            # Verified manually by listening
            (df.ID == "d7fbc4b9-5f94-4639-b82b-6b1592b2ad3a")
            & (df.time_str == "19:58:16")
        )
        | (
            # Verified manually by listening
            (df.ID == "d7fbc4b9-5f94-4639-b82b-6b1592b2ad3a")
            & (df.time_str == "19:55:44")
        )
        | (
            # Verified manually by listening
            (df.ID == "d7fbc4b9-5f94-4639-b82b-6b1592b2ad3a")
            & (df.time_str == "20:00:21")
        )
        | (
            # Verified manually by listening
            (df.ID == "d7fbc4b9-5f94-4639-b82b-6b1592b2ad3a")
            & (df.time_str == "19:59:13")
        )
        | (
            # Verified manually by listening
            (df.ID == "d7fbc4b9-5f94-4639-b82b-6b1592b2ad3a")
            & (df.time_str == "20:01:12")
        )
        # | (
        #     No recording audio in the last 15 seconds
        #     (df["ID"] == "8f42a931-7bd1-4536-bec3-a5f24da3c61f")
        #     & (df["pic_name"] == "Rafd090_36_Caucasian_male_neutral_frontal")
        #     # The Speechbrain VAD is able to deal with this -> so should be fine
        # )
    )
    return bad_recordings


def _silent_end_mask(df: pd.DataFrame) -> pd.Series:
    # no recording audio at the end
    bad_recordings = (
        # A large silent portion at the end
        (df.ID == "63defe6b-db35-4b9c-b53d-90df6612beaa")
        & (df.time_str == "17:48:35")
    ) | (
        # A large silent portion at the end
        (df.ID == "2e98e07d-5d7c-469b-aee1-2421856671b0")
        & (df.time_str == "14:26:20")
    )
    return bad_recordings


def _noisy_recording_mask(df: pd.DataFrame) -> pd.Series:
    # Opensmile + VAD can't deal with this data
    bad_recordings = (
        (
            # A lot of background noise - openSMILE can't properly deal with this data
            (df.ID == "28d01050-e4e0-4115-a5e9-09c8cb917fb0")
            # & (df.time_str == "15:15:20")
        )
        | (  # A loud noise in the middle of the recording
            (df.ID == "099ceb4d-28f3-4b1f-8f8e-16b439a882f1")
            & (df.time_str == "17:06:37")
        )
        | (
            # A lot of noise in the beginning of the recording
            # Verified manually by listening, opensmile cannot process this properly
            (df.ID == "e03516f6-1af7-4c72-ad49-41b6db4733d7")
            & (df.time_str == "22:07:40")
        )
        | (
            # >80% of the recordings are just pure noise for the user below
            (df.ID == "42c842a5-7051-44c3-af42-cf824fded959")
        )
        | (
            # Verified manually by listening
            (df.ID == "d7fbc4b9-5f94-4639-b82b-6b1592b2ad3a")
            & (df.time_str == "19:57:47")
        )
        | (
            # Verified manually by listening
            # A large noise (audio cable disconnect) in the middle of the recording
            (df.ID == "d7fbc4b9-5f94-4639-b82b-6b1592b2ad3a")
            & (df.time_str == "19:52:41")
        )
        | (
            # Verified manually by listening
            # Audio cable disconnect, only reconnect at the end of the recording
            # a lot of noise
            (df.ID == "d7fbc4b9-5f94-4639-b82b-6b1592b2ad3a")
            & (df.time_str == "19:56:41")
        )
        | (
            # Verified manually by listening
            # Audio cable disconnect, only reconnect at the end of the recording
            # a lot of noise
            (df.ID == "d7fbc4b9-5f94-4639-b82b-6b1592b2ad3a")
            & (df.time_str == "19:54:55")
        )
    )

    return bad_recordings


def _skip_id_mask(df: pd.DataFrame) -> pd.Series:
    ids_to_skip = [
        # This user has 14 PiSCES and 10 Radboud recordings under 15 seconds
        # "a2dbee10-fc31-42ec-89f8-9b4e5fa74b7f",  # Too much short sessions
        # Low quality user data
        # TODO -> retain this user
        # "0c28e160-d7bb-4b17-829c-c9ebc7ad0a1f",  # looks okay
        #
        # TODO -> the audio seems okay, however opensmile cannot
        # qualitatively process this
        "3a8266c9-7087-45fe-98e4-6c9c001c0050"
    ]
    return df["ID"].isin(ids_to_skip)


def get_valid_audio_mask(df: pd.DataFrame) -> pd.Series:
    return ~_skip_id_mask(df) & ~_silent_recording_mask(df) & ~_noisy_recording_mask(df)
