# GSSP analysis notebooks


| Notebook | Description |
|------|------|
| [0.1_EDA](0.1_EDA.ipynb) | This notebook withholds some basic Exploratory Data Analysis (EDA) of the `web_app_data` and its corresponding metadata. Specifically: <br><br> - The participant metadata file is parsed <br> -The arousal and valence datafiles are parsed, which are further analyzed in [this notebook](0.2_Arousal_Valence_analysis.ipynb)<br> - Exploratory visualizations of (whole recording) utterance durations, audio sample rate <br> - Parsing of event data  | 
| [0.2_Arousal_Valence](0.2_Arousal_Valence_analysis.ipynb) | **Visual** Analysis of arousal and valence values. Specifically it covers: <br> - arousal & Valence over time <br> - Arousal & valence for each picture stimuli (compared across groups) <br><br> A statistical analysis of the arousal-valence analysis can be found in the [r-script](../scripts/) folder.  |
| **Speech data processing** | |
| [1_transform_audio](0.2.1_Process_audio_Transform.ipynb) | - loads and scales the audio data <br> - resamples audio to 16kHz using pytorch sinc interpolation <br> - scales the data to float32 [0, 1] <br> - save the data as .wav and .npy array   |
| [2_audio_quality_assessment](0.3_Process_audio_Analyze_quality.ipynb) | This notebook contains the visualizations that were utilized to create the [GSSP_analysis](../loc_data/GSSP_manual_analysis.tsv). Furthermore, some additional visualizations are included which focus on single utterances (afhter manual inspection was completed). |
| [3_VAD_slicing](0.4_Process_audio_Parse_VAD_slice.ipynb) | Applies an open-access (speechbrain/huggignface) Voice Activity Detection model to find the outer voiced bounds of each segment. Afterwards, these bounds are padded with a margin to ensure that the segments  |
| **Speech feature extraction** | |
| [5_OpenSMILE feature extraction](0.4.1_Process_audio_Parse_Extract_feats.ipynb) |  Perform (fixed duration) feature extraction on the 16kHZ VAD-sliced audio segments |
| **Speech analysis** | |
| [OpenSMile Visualization](0.5_OpenSMILE_visualizations.ipynb) |  Visualizes the fixed duration OpenSMILE GeMAPSv01b functional features w.r.t. Speech acquisition task. |
| [ECAPA-TDNN](0.6_ECPA_TDNN_npy.ipynb) | Extract and projects (using t-SNE) ECAPA-TDNN embeddings from fixed and whole duration utterances.<br><br>A machine learning model is also utilized to assess the speech style separability of the embeddings.  |
| **External validation** | |
| [CGN Parsing](1.1_CGN_EDA_parsing.ipynb) | This notebook parses the [Corpus Gesproken Nederlands](https://ivdnt.org/images/stories/producten/documentatie/cgn_website/doc_English/topics/index.htm) (CGN) its orthographic description and speaker recording. |
| [CGN Feature extraction](1.2_CGN_listen_extract_feats.ipynb) | This notebook allows listening to excerpts of the selected CGN components and extracts the OpenSMILE GeMAPSv01b Functional features.|
| [OpenSmile ML](1.3_OpenSMILE_ML.ipynb) | This notebooks performs a within GSSP web app dataset validation utilizing the OpenSMILE GeMAPSv01b functional features.<br><br>Afterwards, a subset of these functional features are selected to train on the whole web app dataset and predict on the `CGN` dataset, hinting for speech style generalizability.|
