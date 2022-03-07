from pathlib import Path

import numpy as np
import json

from visualization import visualize_frames
from sh_kinematics import shadow_hand_fk


if __name__ == "__main__":
    path = Path(__file__).parent / "saves" / "eigengrasps.json"
    with open(path, "r") as f:
        eigengrasps = json.load(f)
    visualize_frames(shadow_hand_fk(np.concatenate((np.zeros(3),np.array(eigengrasps["2"]["joints"])))*-5))