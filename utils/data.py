from pathlib import Path
from typing import Dict

from data import load_data


def load_data(path_audio: Path, path_labels: Path, codes: Dict) -> Dict:
    files_tr = load_data(path_audio, codes["training"])
    labels_tr = load_data(path_labels, codes["training"]).astype(float)
    files_val = load_data(path_audio, codes["validation"])
    labels_val = load_data(path_labels, codes["validation"]).astype(float)
    files_test = load_data(path_audio, codes["testing"])
    labels_test = load_data(path_labels, codes["testing"]).astype(float)
    data = {
        "training": {
            "files": files_tr,
            "labesls": labels_tr,
        },
        "validation": {
            "files": files_val,
            "labesls": labels_val,
        },
        "testing": {
            "files": files_test,
            "labesls": labels_test,
        },
    }
    return data
