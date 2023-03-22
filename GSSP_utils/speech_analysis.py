from typing import List, Optional

import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import opensmile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torchaudio
from IPython import display
from IPython.lib.display import Audio
from ipywidgets import GridspecLayout, Output
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from speechbrain.pretrained.interfaces import VAD

from .path_conf import interim_speech_data_dir, speech_data_session_dir

# Load the VAD model
VAD_model = VAD.from_hparams(
    source="speechbrain/vad-crdnn-libriparty", savedir=".vad_model"
)


def whole_duration_image(
    r: pd.Series,
    feat_cols: List[str],
    df_smile_lld,
    vad: bool = True,
    plot_type: str = "png",
):
    """Plot the whole duration of the recording together with the VAD output and the
    opensmile LLD features.
    """
    df_smile_lld_utt = df_smile_lld[
        (df_smile_lld.ID == r.ID)
        & (df_smile_lld.time_str == r.time_str)
        # & (df_gemaps_lld.DB == r.DB)
    ]
    assert len(df_smile_lld_utt) > 0

    arr_16khz_n = np.load(
        df_smile_lld_utt.iloc[0].file.split(".wav")[0] + ".npy"
    ).ravel()
    t_arr_n = np.arange(len(arr_16khz_n)) / 16_000

    e2e_boundaries_tuning = None
    if vad:
        e2e_boundaries_tuning = (
            VAD_model.upsample_boundaries(
                VAD_model.get_speech_segments(
                    audio_file=str(df_smile_lld_utt.iloc[0].file),
                    large_chunk_size=15,
                    small_chunk_size=1,
                    overlap_small_chunk=True,
                    apply_energy_VAD=True,
                    double_check=True,
                    # neural proba activation thresholds
                    activation_th=0.65,
                    deactivation_th=0.2,
                    # VAD energy activation thresholds
                    en_activation_th=0.6,
                    en_deactivation_th=0.3,
                ),
                audio_file=str(df_smile_lld_utt.iloc[0].file),
            )
            .numpy()
            .ravel()
        )

    n_rows = 1 + len(feat_cols)
    subplot_kwargs = {}
    if n_rows > 1:
        subplot_kwargs["vertical_spacing"] = 0.05

    fr = FigureResampler(
        make_subplots(
            rows=n_rows,
            shared_xaxes=True,
            **subplot_kwargs,
            specs=[[{"secondary_y": True}]] * n_rows,
        ),
        default_n_shown_samples=2500,
    )
    # Row 1: normalized wav + 16Khz resampled wav
    fr.add_trace(
        go.Scattergl(name="torch-norm"),  # "opacity": 0.5},
        hf_y=arr_16khz_n,
        hf_x=t_arr_n,
        max_n_samples=10_000 if plot_type in ["png", "return"] else 2500,
        col=1,
        row=1,
    )

    if e2e_boundaries_tuning is not None:
        fr.add_trace(
            go.Scattergl(name="VAD boundaries"),
            hf_y=e2e_boundaries_tuning,
            hf_x=t_arr_n,
            secondary_y=True,
            col=1,
            row=1,
        )

        # Add a rectangle where we would cut the audio
        where = np.where(e2e_boundaries_tuning > 0)[0]
        if not len(where):
            speech_start_idx = 0
            speech_end_idx = len(e2e_boundaries_tuning) - 1
        else:
            speech_start_idx, speech_end_idx = where[0], where[-1]
        fr.add_vrect(
            x0=0,
            x1=max(0, t_arr_n[speech_start_idx] - 0.25),
            line_width=0,
            fillcolor="red",
            opacity=0.2,
        )
        fr.add_vrect(
            x0=min(t_arr_n[-1], t_arr_n[speech_end_idx] + 0.25),
            x1=t_arr_n[-1],
            line_width=0,
            fillcolor="red",
            opacity=0.2,
        )

    # Add the feature columns
    for i, col in enumerate(feat_cols, start=2):
        fr.add_trace(
            go.Scattergl(name=col),
            hf_y=df_smile_lld_utt[col].values,
            hf_x=df_smile_lld_utt["end"],
            max_n_samples=10_000 if plot_type == "png" else 2500,
            col=1,
            row=i,
        )

    # update layout and show
    fr.update_layout(
        height=200 + 150 * n_rows,
        title=f"{r.ID} - <b>{r.DB}</b> - {r.pic_name}__{r.time_str}",
        title_x=0.5,
        template="plotly_white",
    )

    if plot_type == "png":
        fr.show(renderer="png", width=2000, height=250 + 200 * n_rows)
    elif plot_type == "dash":
        fr.show_dash(mode="inline", port=8034)
    elif plot_type == "return":
        return fr
    # Default: show the plot
    fr.show()


def analyze_audio_quality(
    df_session: pd.DataFrame,
    ID: str,
    n_marloes: int = 2,
    n_pisces: int = 2,
    n_radboud: int = 2,
    smile_lld: opensmile.Smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    ),
    feat_cols: List[str] = ["F0semitoneFrom27.5Hz_sma3nz", "jitterLocal_sma3nz"],
    audio=True,
    audio_begin_s=10,
    audio_end_s=15,
    show_wav_features=False,
    plot=True,
    plot_type="png",
    norm_audio=True,
    vad=True,
    noise_reduction=True,
    random_state=42,
) -> None:
    """Analyze audio quality of a single user."""
    # Gather the correct dataframe slices
    global user_df, wav_path_orig
    user_df = df_session[df_session["ID"] == ID].copy()
    df_u_marloes = user_df[user_df.DB == "marloes"].copy()
    df_u_pisces = user_df[user_df.DB == "PiSCES"].copy()
    df_u_radboud = user_df[user_df.DB == "Radboud"].copy()

    # Print user statistics
    print(
        f"age-sex: {user_df.iloc[0]['age']} - {user_df.iloc[0].sex}\t\t"
        + f"education: {user_df.iloc[0]['education']}\t\t"
        + f"device: {user_df.iloc[0]['device']}"
    )
    print(
        f"#Radboud: {len(df_u_radboud)}   -   #Pisces: {len(df_u_pisces)}   -   "
        + f"#Marloes: {len(df_u_marloes)}  -  SR={user_df.iloc[0].wav_sample_rate}Hz"
    )

    # Visualize the duration of the recordings
    sns.catplot(
        data=user_df,
        x="wav_duration_s",
        y="DB",
        height=3,
        aspect=5,
        s=15,
        palette="Set2",
        alpha=0.5,
    )
    plt.show()
    plt.close()

    # Calculate the correlation
    corr_list = []
    for n, df_u_db in [
        (n_marloes, df_u_marloes),
        (n_pisces, df_u_pisces),
        (n_radboud, df_u_radboud),
    ]:
        if n > 0:
            print("=" * 30, df_u_db.iloc[0]["DB"], "=" * 30)

        if len(df_u_db) < n:
            print(f"Warning: {ID} has less than {n} {df_u_db.DB.iloc[0]} recordings")

        slc = df_u_db.sample(n, random_state=random_state)
        # print(f"Selected {slc.shape[0]} recordings from {df_u_db.DB.iloc[0]}")
        for _, row in slc.iterrows():
            # ------------------ Load audio data ------------------ #
            ## The raw, 16 bit PCM wav path
            wav_path_orig = list(
                speech_data_session_dir.glob(
                    f"*{row.ID}/{row.DB}/{row.pic_name}__{row.time_str}.wav"
                )
            )[0]
            arr_orig_wav_n, fs_orig = torchaudio.load(wav_path_orig, normalize=True)
            arr_orig_wav_n = arr_orig_wav_n.numpy().ravel()

            ## THe 16khz, 32 bit float wav path
            wav_path_16khz = list(
                (interim_speech_data_dir / "full_dur_16khz").glob(
                    f"*{row.ID}/{row.DB}/{row.pic_name}__{row.time_str}.wav"
                )
            )[0]
            arr_16khz_n, fs = (
                np.load(wav_path_16khz.parent / (wav_path_16khz.stem + ".npy")).ravel(),
                16_000,
            )
            t_arr_n = np.arange(0, arr_16khz_n.shape[0]) / fs

            arr_n_16Khz_nrs_v2 = None
            if noise_reduction:  # Noise reduction
                # ORIGv1
                # arr_n_16Khz_nrs = nr.reduce_noise(
                #     arr_16khz_n,
                #     sr=16_000,
                #     stationary=True,
                #     n_fft=256,
                #     n_std_thresh_stationary=0.75,
                #     prop_decrease=0.95,
                #     time_mask_smooth_ms=25,
                # )
                arr_n_16Khz_nrs_v2 = nr.reduce_noise(
                    arr_16khz_n,
                    sr=16_000,
                    stationary=True,
                    n_std_thresh_stationary=0.75,
                    n_fft=256,
                    prop_decrease=0.85,
                    time_mask_smooth_ms=25,
                )

            audio_list = []
            if audio:  # Listen to audio
                start, stop = audio_begin_s, audio_end_s
                audio_list.append(
                    (
                        "orig norm",
                        arr_orig_wav_n[start * fs_orig : stop * fs_orig],
                        fs_orig,
                    )
                )
                if norm_audio:
                    audio_list.append(
                        (
                            "16khz norm",
                            arr_16khz_n[start * 16_000 : stop * 16_000],
                            16_000,
                        )
                    )
                if arr_n_16Khz_nrs_v2 is not None:  # noise reduction
                    audio_list.append(
                        (
                            "16khz NR v2",
                            arr_n_16Khz_nrs_v2[start * 16_000 : stop * 16_000],
                            16_000,
                        )
                    )
                print(
                    f"Playing from {start} to {stop} seconds  |"
                    + "| ".join([a[0] for a in audio_list])
                    + " |"
                )

                if len(audio_list):
                    grid = GridspecLayout(1, len(audio_list))
                    for col_idx, (name, arr, fs) in enumerate(audio_list):
                        out = Output()
                        try:
                            with out:
                                display.display(Audio(arr, rate=fs))
                            grid[0, col_idx] = out
                        except KeyError:
                            continue
                    display.display(grid)

            # Extract opensmile features
            df_smile_arr_orig_n = smile_lld.process_signal(arr_orig_wav_n, fs_orig)
            reference = df_smile_arr_orig_n
            df_smile_wav, df_smile_arr_n_16Khz_nrs_v2 = None, None
            if show_wav_features:
                df_smile_wav = smile_lld.process_file(str(wav_path_orig))
                reference = df_smile_wav
            df_smile_arr_n_16Khz = smile_lld.process_signal(arr_16khz_n, 16_000)
            if arr_n_16Khz_nrs_v2 is not None:
                df_smile_arr_n_16Khz_nrs_v2 = smile_lld.process_signal(
                    arr_n_16Khz_nrs_v2, 16_000
                )

            # Visualize autocorrelation
            corr_list += [
                pd.Series(
                    {
                        feat_col: np.round(
                            np.corrcoef(reference[feat_col], df_comp[feat_col]), 3
                        )[0, 1]
                        for feat_col in df_smile_arr_orig_n.columns
                    },
                    name=name + f"__{row.DB}_{row.DB_no}_{row.session_no}",
                ).to_frame()
                for name, df_comp in [("norm_16Khz", df_smile_arr_n_16Khz)]
                + (
                    [
                        ("norm_16Khz_nrs_v2", df_smile_arr_n_16Khz_nrs_v2),
                    ]
                    if df_smile_arr_n_16Khz_nrs_v2 is not None
                    else []
                )
            ]

            e2e_boundaries_tuning = None
            if vad:
                e2e_boundaries_tuning = (
                    VAD_model.upsample_boundaries(
                        VAD_model.get_speech_segments(
                            audio_file=str(wav_path_16khz),
                            large_chunk_size=15,
                            small_chunk_size=1,
                            overlap_small_chunk=True,
                            apply_energy_VAD=True,
                            double_check=True,
                            # neural proba activation thresholds
                            activation_th=0.65,
                            deactivation_th=0.2,
                            # VAD energy activation thresholds
                            en_activation_th=0.6,
                            en_deactivation_th=0.3,
                        ),
                        audio_file=str(wav_path_16khz),
                    )
                    .numpy()
                    .ravel()
                )

            # Plot
            n_rows = 1 + len(feat_cols)
            if plot:
                fr = FigureResampler(
                    make_subplots(
                        rows=n_rows,
                        shared_xaxes=True,
                        # vertical_spacing=0.05,
                        specs=[[{"secondary_y": True}]] * n_rows,
                    ),
                    default_n_shown_samples=2500,
                )
                # Row 1: normalized wav + 16Khz resampled wav
                fr.add_trace(
                    go.Scattergl(name="torch-norm"),  # "opacity": 0.5},
                    hf_y=arr_16khz_n,
                    hf_x=t_arr_n,
                    max_n_samples=10_000,
                    col=1,
                    row=1,
                )

                if e2e_boundaries_tuning is not None:
                    fr.add_trace(
                        go.Scattergl(name="VAD boundaries"),
                        hf_y=e2e_boundaries_tuning,
                        hf_x=t_arr_n,
                        secondary_y=True,
                        col=1,
                        row=1,
                    )

                    # Add a rectangle where we would cut the audio
                    where = np.where(e2e_boundaries_tuning > 0)[0]
                    if not len(where):
                        speech_start_idx, speech_end_idx = (
                            0,
                            len(e2e_boundaries_tuning) - 1,
                        )
                    else:
                        speech_start_idx, speech_end_idx = where[0], where[-1]
                    fr.add_vrect(
                        x0=0,
                        x1=max(0, t_arr_n[speech_start_idx] - 0.25),
                        line_width=0,
                        fillcolor="red",
                        opacity=0.2,
                    )
                    fr.add_vrect(
                        x0=min(t_arr_n[-1], t_arr_n[speech_end_idx] + 0.25),
                        x1=t_arr_n[-1],
                        line_width=0,
                        fillcolor="red",
                        opacity=0.2,
                    )

                # fr.add_trace(
                #     {"name": "norm-16Khz"},  # "opacity": 0.5},
                #     hf_y=arr_n_16Khz,
                #     hf_x=np.arange(len(arr_n_16Khz)) / 16_000,
                #     col=1,
                #     row=1,
                # )

                c_list = px.colors.qualitative.Plotly
                signal_names = []
                for i, feat_col in enumerate(feat_cols, start=2):
                    fr.add_annotation(
                        xref="x domain",
                        yref="y domain",
                        x=0.5,
                        y=1.1,
                        showarrow=False,
                        text=f"<b>{feat_col}</b>",
                        row=i,
                        col=1,
                    )

                    if df_smile_wav is not None:
                        fr.add_trace(
                            {
                                "name": "Smile-orig-wav",
                                "line_color": c_list[2],
                                "showlegend": "opensmile-wav" not in signal_names,
                                "legendgroup": "opensmile-wav",
                            },
                            hf_y=df_smile_wav[feat_col],
                            hf_x=df_smile_wav.reset_index().start.dt.total_seconds(),
                            col=1,
                            row=i,
                        )
                        signal_names.append("opensmile-wav")

                    # Visualize the opensmile features
                    for color_idx, (name, df) in enumerate(
                        [
                            ("Smile-orig-n", df_smile_arr_orig_n),
                            ("Smile-16khz-n", df_smile_arr_n_16Khz),
                        ]
                        + (
                            [
                                ("norm_16Khz_nrs_v2", df_smile_arr_n_16Khz_nrs_v2),
                            ]
                            if noise_reduction
                            else []
                        ),
                        start=2 + int(show_wav_features),
                    ):
                        fr.add_trace(
                            {
                                "name": name,  #  + '-' + feat_col,
                                "line_color": c_list[color_idx],
                                "showlegend": name not in signal_names,
                                "legendgroup": name,
                            },
                            hf_y=df[feat_col],
                            hf_x=df.reset_index().start.dt.total_seconds(),
                            col=1,
                            row=i,
                        )
                        signal_names.append(name)

                # update layout and show
                fr.update_layout(
                    # legend=dict(
                    #     orientation="h",
                    #     y=1.06, xanchor="left", x=0, font=dict(size=18)
                    # ),
                    height=150 + 200 * n_rows,
                    # height=800,
                    title=(
                        f"{row.ID} - <b>{row.DB}</b> - "
                        f"{row.pic_name}__{row.time_str}"
                    ),
                    title_x=0.5,
                    # colorway=px.colors.qualitative.Dark2,
                    template="plotly_white",
                )
                if plot_type == "png":
                    fr.show(renderer="png", width=1400)
                elif plot_type == "dash":
                    fr.show_dash(mode="inline", port=8034)
                else:
                    fr.show()

    # Display the correlation results
    cm = sns.light_palette("red", reverse=True, as_cmap=True)
    display.display(
        pd.concat(corr_list, axis=1).style.background_gradient(cmap=cm, vmin=0, vmax=1)
    )


def analyze_utterance(
    utterance: pd.Series,
    smile_lld: Optional[opensmile.Smile] = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    ),
    # feat_cols: List[str] = ["F0semitoneFrom27.5Hz_sma3nz", "jitterLocal_sma3nz"],
    feat_cols: List[str] = ["F0semitoneFrom27.5Hz_sma3nz", "shimmerLocaldB_sma3nz"],
    # Listen to audio
    audio=True,
    audio_begin_s=10,
    audio_end_s=15,
    # Visualize audio (opensmile)
    plot=True,
    plot_type="png",
    # Transform audio
    norm_audio=True,
    noise_reduction=True,
    noise_factor=0,
    show_noise_audio=False,
    vad=True,
    show_corr=True,
) -> None:
    """Analyze audio quality of a single File."""

    # Print user statistics
    print(
        f"age-sex: {utterance['age']} - {utterance.sex}\t\t"
        + f"education: {utterance['education']}\t\t"
        + f"device: {utterance['device']}"
    )

    # Calculate the correlation
    corr_list = []

    # ------------------ Load audio data ------------------ #
    ## The raw, 16 bit PCM wav path
    wav_path_orig = list(
        speech_data_session_dir.glob(
            f"*{utterance.ID}/{utterance.DB}/{utterance.pic_name}__{utterance.time_str}.wav"
        )
    )[0]
    arr_orig_wav_n, fs_orig = torchaudio.load(wav_path_orig, normalize=True)
    arr_orig_wav_n = arr_orig_wav_n.numpy().ravel()
    print('range', round(arr_orig_wav_n.min(), 2), round(arr_orig_wav_n.max(), 2), round(np.ptp(arr_orig_wav_n), 2))
    print('range', np.round(np.quantile(arr_orig_wav_n, [0.001, 0.999]), 2))
    # min_ampl, max_ampl  = np.quantile(arr_orig_wav_n, [0.002, 0.998]), 2)


    ## THe 16khz, 32 bit float wav path
    wav_path_16khz = list(
        (interim_speech_data_dir / "full_dur_16khz").glob(
            f"*{utterance.ID}/{utterance.DB}/{utterance.pic_name}__{utterance.time_str}.wav"
        )
    )[0]
    arr_16khz_n, fs = (
        np.load(wav_path_16khz.parent / (wav_path_16khz.stem + ".npy")).ravel(),
        16_000,
    )
    t_arr_n = np.arange(0, arr_16khz_n.shape[0]) / fs

    # TODO: de noise factor gaan relaten aan gemiddelde energie van de audio
    # dit dan in DB gaan uitdrukken en zo formuleren als step
    # ... in order to mitigate OpenSMILEs
    arr_16khz_n_noisy = None
    arr_orig_wav_n_noisy = None
    if noise_factor > 0:
        rand_func = np.random.rand
        noise = (rand_func(len(arr_16khz_n)) * noise_factor).astype(np.float32)
        arr_16khz_n_noisy = arr_16khz_n + noise
        noise = (rand_func(len(arr_orig_wav_n)) * noise_factor).astype(np.float32)
        arr_orig_wav_n_noisy = arr_orig_wav_n + noise
    t_arr_orig_n = np.arange(arr_orig_wav_n.shape[0]) / fs_orig

    arr_n_16Khz_nrs_v2 = None
    if noise_reduction:  # Noise reduction
        arr_n_16Khz_nrs_v2 = nr.reduce_noise(
            arr_16khz_n,
            sr=16_000,
            stationary=True,
            n_std_thresh_stationary=0.75,
            n_fft=256,
            prop_decrease=0.85,
            time_mask_smooth_ms=25,
        )

    audio_list = []
    if audio:  # Listen to audio
        start, stop = audio_begin_s, audio_end_s
        print(f"Playing from {start} to {stop} seconds")
        audio_list.append(
            (
                "orig norm",
                arr_orig_wav_n[start * fs_orig : stop * fs_orig],
                fs_orig,
            )
        )
        if norm_audio:
            audio_list.append(
                (
                    "16khz norm",
                    arr_16khz_n[start * 16_000 : stop * 16_000],
                    16_000,
                )
            )
        if arr_n_16Khz_nrs_v2 is not None:  # noise reduction
            audio_list.append(
                (
                    "16khz NR v2",
                    arr_n_16Khz_nrs_v2[start * 16_000 : stop * 16_000],
                    16_000,
                )
            )

    if len(audio_list):
        grid = GridspecLayout(2, len(audio_list))
        for col_idx, (name, arr, fs) in enumerate(audio_list):
            out = Output()
            try:
                with out:
                    display.display(Audio(arr, rate=fs))
                grid[1, col_idx] = out
            except KeyError:
                continue
            out = Output()
            with out:
                display.display(name)
            grid[0, col_idx] = out

        display.display(grid)

    # Extract opensmile features
    if smile_lld is not None:
        df_smile_arr_orig_n = smile_lld.process_signal(arr_orig_wav_n, fs_orig)
        reference = df_smile_arr_orig_n
        df_smile_wav, df_smile_arr_n_16Khz_nrs_v2 = None, None
        df_smile_arr_n_16Khz = smile_lld.process_signal(arr_16khz_n, 16_000)
        if arr_n_16Khz_nrs_v2 is not None:
            df_smile_arr_n_16Khz_nrs_v2 = smile_lld.process_signal(
                arr_n_16Khz_nrs_v2, 16_000
            )
        df_smile_arr_n_16Khz_noise = None
        df_smile_arr_n_orig_noise = None
        if noise_factor > 0:
            df_smile_arr_n_16Khz_noise = smile_lld.process_signal(
                arr_16khz_n_noisy, 16_000
            )
            df_smile_arr_n_orig_noise = smile_lld.process_signal(
                arr_orig_wav_n_noisy, fs_orig
            )

    # Visualize autocorrelation
    if show_corr:
        corr_list += [
            pd.Series(
                {
                    feat_col: np.round(
                        np.corrcoef(reference[feat_col], df_comp[feat_col]), 3
                    )[0, 1]
                    for feat_col in df_smile_arr_orig_n.columns
                },
                name=name
                + f"__{utterance.DB}_{utterance.DB_no}_{utterance.session_no}",
            ).to_frame()
            for name, df_comp in [("norm_16Khz", df_smile_arr_n_16Khz)]
            + (
                [
                    ("norm_16Khz_nrs_v2", df_smile_arr_n_16Khz_nrs_v2),
                ]
                if df_smile_arr_n_16Khz_nrs_v2 is not None
                else []
            )
        ]

    e2e_boundaries_tuning = None
    if vad:
        e2e_boundaries_tuning = (
            VAD_model.upsample_boundaries(
                VAD_model.get_speech_segments(
                    audio_file=str(wav_path_16khz),
                    large_chunk_size=15,
                    small_chunk_size=1,
                    overlap_small_chunk=True,
                    apply_energy_VAD=True,
                    double_check=True,
                    # neural proba activation thresholds
                    activation_th=0.65,
                    deactivation_th=0.2,
                    # VAD energy activation thresholds
                    en_activation_th=0.6,
                    en_deactivation_th=0.3,
                ),
                audio_file=str(wav_path_16khz),
            )
            .numpy()
            .ravel()
        )

    # Plot
    n_rows = 1 + len(feat_cols)
    subplot_kwargs = {}
    if n_rows > 1:
        subplot_kwargs["vertical_spacing"] = 0.05
    if plot:
        fr = FigureResampler(
            make_subplots(
                rows=n_rows,
                shared_xaxes=True,
                **subplot_kwargs,
                specs=[[{"secondary_y": True}]] * n_rows,
            ),
            default_n_shown_samples=2500,
        )
        # Row 1: normalized wav + 16Khz resampled wav
        fr.add_trace(
            go.Scattergl(name="torch-norm"),  # "opacity": 0.5},
            hf_y=arr_16khz_n,
            hf_x=t_arr_n,
            max_n_samples=10_000 if plot_type == "png" else 2500,
            col=1,
            row=1,
        )
        if noise_factor > 0 and show_noise_audio:
            fr.add_trace(
                go.Scattergl(name="torch-norm"),  # "opacity": 0.5},
                hf_y=arr_orig_wav_n_noisy,
                hf_x=t_arr_orig_n,
                max_n_samples=10_000 if plot_type in ["png", "return", "plotly"] else 2500,
                col=1,
                row=1,
            )

        if e2e_boundaries_tuning is not None:
            fr.add_trace(
                go.Scattergl(name="VAD boundaries"),
                hf_y=e2e_boundaries_tuning,
                hf_x=t_arr_n,
                secondary_y=True,
                col=1,
                row=1,
            )

            # Add a rectangle where we would cut the audio
            where = np.where(e2e_boundaries_tuning > 0)[0]
            if not len(where):
                speech_start_idx, speech_end_idx = (
                    0,
                    len(e2e_boundaries_tuning) - 1,
                )
            else:
                speech_start_idx, speech_end_idx = where[0], where[-1]
            fr.add_vrect(
                x0=0,
                x1=max(0, t_arr_n[speech_start_idx] - 0.25),
                line_width=0,
                fillcolor="red",
                opacity=0.2,
            )
            fr.add_vrect(
                x0=min(t_arr_n[-1], t_arr_n[speech_end_idx] + 0.25),
                x1=t_arr_n[-1],
                line_width=0,
                fillcolor="red",
                opacity=0.2,
            )

        c_list = px.colors.qualitative.Plotly
        signal_names = []
        for i, feat_col in enumerate(feat_cols, start=2):
            fr.add_annotation(
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=1.1,
                showarrow=False,
                text=f"<b>{feat_col}</b>",
                row=i,
                col=1,
            )

            if df_smile_wav is not None:
                fr.add_trace(
                    {
                        "name": "Smile-orig-wav",
                        "line_color": c_list[2],
                        "showlegend": "opensmile-wav" not in signal_names,
                        "legendgroup": "opensmile-wav",
                    },
                    hf_y=df_smile_wav[feat_col],
                    hf_x=df_smile_wav.reset_index().start.dt.total_seconds(),
                    col=1,
                    row=i,
                )
                signal_names.append("opensmile-wav")

            # Visualize the opensmile features
            for color_idx, (name, df) in enumerate(
                [
                    ("Smile-orig-n", df_smile_arr_orig_n),
                    ("Smile-16khz-n", df_smile_arr_n_16Khz),
                    *(
                        [
                            ("Smile-16khz-noise", df_smile_arr_n_16Khz_noise),
                            ("Smile-orig-n-noise", df_smile_arr_n_orig_noise),
                        ]
                        if noise_factor > 0
                        else []
                    ),
                ]
                + (
                    [
                        ("norm_16Khz_nrs_v2", df_smile_arr_n_16Khz_nrs_v2),
                    ]
                    if noise_reduction
                    else []
                ),
                start=2,
            ):
                fr.add_trace(
                    {
                        "name": name,  #  + '-' + feat_col,
                        "line_color": c_list[color_idx],
                        "showlegend": name not in signal_names,
                        "legendgroup": name,
                    },
                    hf_y=df[feat_col],
                    hf_x=df.reset_index().start.dt.total_seconds(),
                    col=1,
                    row=i,
                )
                signal_names.append(name)

        # update layout and show
        fr.update_layout(
            height=200 + 150 * n_rows,
            title=(
                f"{utterance.ID} - <b>{utterance.DB}</b> - "
                f"{utterance.pic_name}__{utterance.time_str}"
            ),
            title_x=0.5,
            template="plotly_white",
        )
        fr.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
        if plot_type == "png":
            fr.show(renderer="png", width=1400)
        elif plot_type == "dash":
            fr.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=.95,
                )
            )
            fr.show_dash(mode="inline", port=8025)
        elif plot_type == "return":
            return fr
        else:
            fr.show()

    if show_corr:
        cm = sns.light_palette("red", reverse=True, as_cmap=True)
        display.display(
            pd.concat(corr_list, axis=1).style.background_gradient(
                cmap=cm, vmin=0, vmax=1
            )
        )
