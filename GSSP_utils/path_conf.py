from datetime import datetime
from inspect import getsourcefile
from pathlib import Path
from socket import gethostname

if gethostname() == "gecko":
    data_dir = Path("/media/SPS/")  # this directory needs to be adjusted
    speech_data_root_dir = data_dir / "web_app_data"

    # Path towards the "Corpus Gesproken Nederlands" data
    cgn_root_dir = Path("/media/SPS/cgn/")
    cgn_ort_path = cgn_root_dir / "ort"
else:
    raise ValueError(
        "Unknow user, please comment this line and add your own path below"
    )
    # the path to the osf `speech_web_app` data directory - https://osf.io/8wrmn
    data_dir = Path("TODO")
    speech_data_root_dir = data_dir / "web_app_data"

    # If you want to use the CGN data, you need to download it yourself
    # and add the path to the directory here
    cgn_root_dir = Path("TODO")
    cgn_ort_path = cgn_root_dir / "ort"

speech_data_session_dir = speech_data_root_dir.joinpath("backup")
speech_web_app_image_dir = speech_data_root_dir.joinpath("img")

# Where intermediate data files are stored
interim_dir = data_dir / "interim/"
interim_cgn_dir = interim_dir / "cgn"
interim_speech_data_dir = interim_dir / "speech_webapp"

# Local pandas dataframes
loc_data_dir = Path(getsourcefile(lambda: 0)).parent.parent.absolute() / "loc_data"


# The date on which users were recuited via the prolific platform
# Might be useful for future analysis
prolific_date = datetime(2022, 6, 22)
