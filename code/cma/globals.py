from pathlib import Path

import yaml

with open("../globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(DATA_DIR, MODEL_DIR, REMOTE_ROOT_URL) = (
    Path(z)
    for z in [
        data["DATA_DIR"],
        data['MODEL_DIR'],
        data['REMOTE_ROOT_URL']

    ]
)
