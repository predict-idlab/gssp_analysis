from datetime import datetime
from pathlib import Path
from inspect import getsourcefile

data_dir = Path('/media') # this directory needs to be adjusted
speech_data_root_dir = data_dir / 'speech_webapp_cleaned'

speech_data_session_dir = speech_data_root_dir.joinpath('backup')
speech_web_app_image_dir = speech_data_root_dir.joinpath('img')

# Where intermediate data files are stored
interim_dir = data_dir / 'interim/'
interim_cgn_dir = interim_dir / 'cgn'
interim_speech_data_dir = interim_dir / 'speech_webapp'

# Local pandas dataframes
loc_data_dir = Path(getsourcefile(lambda:0)).parent.parent.absolute() / 'loc_data'

# Path towards the "Corpus Gesproken Nederlands" data
cgn_root_dir = Path('/media/cgn/')
cgn_ort_path = cgn_root_dir / 'ort'


# The date on which users were recuited via the prolific platform
prolific_date = datetime(2022, 6, 22)