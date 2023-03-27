# GSSP analysis notebooks


| Notebook | Description |
|------|------|
| [0.1_EDA](0.1_EDA.ipynb) | This notebook witholds some basic Exploratory Data Analysis (EDA) of the `web_app_data` and its corresponding metadata. Specifically: <br><br> - The participant metadata file is parsed <br> -The arousal and valence datafiles are parsed, which are further analyzed in [this notebook](0.2_Arousal_Valence_analysis.ipynb)<br> - Exploratory visualizations of (whole recording) utterance durations, audio sample rate <br> - Parsing of event data  | 
| [0.2_Arousal_Valence](0.2_Arousal_Valence_analysis.ipynb) | **Visual** Analysis of arousal and valence values. Specifically it covers: <br> - arousal & Valence over time <br> - Arousal & valence for each picture stimuli (compared accrous groups) <br><br> A statistical analysis of the arousal-valence analysis can be found in the [r-script](../scripts/) folder.  |
| **Speech data processing** | |
| [1_transform_audio](0.2.1_Process_audio_Transform.ipynb) | - loads and scales the audio data <br> - resamples audio to 16kHz using pytorch sinc interpolation <br> - scales the data to float32 [0, 1] <br> - save the data as .wav an npy array   |
| [2_audio_quality_assessment](0.3_Process_audio_Analyze_quality.ipynb) | This notebook contains the visualizations that were utilized to create the [GSSP_analysis](../loc_data/GSSP_manual_analysis.tsv). Futhermore, some additional visualizations are included which focus on single utterances (afhter manual inspection was completed). |
| [3_VAD_slicing](0.4_Process_audio_Parse_VAD_slice.ipynb) | Applies an open-access (speechbrain/huggignface) Voice Activity Detectio model to find the outer voiced bounds of each segment. Afterwards, these bounds are padded with a margin to ensure that the segments  |
| **Speech feature extraction** | |
| [5_OpenSMILE feature extraction](0.4.1_Process_audio_Parse_Extract_feats.ipynb) |  Perform (fixed duration) feature extraction on the 16kHZ VAD-sliced audio semgents|
| **Speech analysis** | |
| [ECAPA-TDNN](0.6_ECPA_TDNN_npy.ipynb) | Extract and projects (using t-SNE) ECAPA-TDNN embeddings from fixed and whole duration utterances.<br><br>A machine learning model is also utilized to assess the speech style seperability of the embeddings.  | 


| []()