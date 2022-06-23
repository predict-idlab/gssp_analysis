from pathlib import Path

# speech_data_root_dir = Path('/speechdump/jonvdrdo/data/speech_web_app')
# speech_data_root_dir = Path('/media/speech_webapp')
# speech_data_root_dir = Path('/users/jonvdrdo/jonas/data/speech_webapp')
speech_data_root_dir = Path('/project_scratch/data/speech_web_app')

speech_data_session_dir = speech_data_root_dir.joinpath('backup')

speech_web_app_image_dir = speech_data_root_dir.joinpath('img')

loc_data_dir = Path('../loc_data/')
