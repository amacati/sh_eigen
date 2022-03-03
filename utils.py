import logging
from pathlib import Path

import numpy as np
import json
from tqdm import tqdm

from tf import zrot_matrix



logger = logging.getLogger(__name__)

def load_dataset(path: Path):
    cache_path = Path(__file__).parent / ".cache" / "dataset_cache.json"
    if cache_path.exists():
        logger.info("Loading data from cache")
        try:
            with open(cache_path, "r") as f:
                joint_list = json.load(f)
            return [np.array(joints) for joints in joint_list]
        except json.JSONDecodeError:
            logger.warning("Corrupted cache read, reloading dataset from source")
            pass
    data_path = path / "data" / "contactpose_data"
    files = [p for p in data_path.glob("**/annotations.json")]
    if not len(files):
        raise RuntimeError((f"Path {path} contains no data samples. Make sure the path points to "
                             "the root of the ContactPose repository and you downloaded the "
                             "dataset"))
    joint_list = []
    progress_bar = tqdm(total=len(files), desc="Loading files", position=0, leave=False)
    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)
        if data["hands"][1]["valid"]:
            joints = _normalize_joints(np.array(data["hands"][1]["joints"]))
            joint_list.append(joints)
            progress_bar.update()
        else:
            progress_bar.total -= 1
            progress_bar.refresh()
    logger.info("Saving files to cache")
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "w+") as f:
        json.dump([joints.tolist() for joints in joint_list], f)
    return joint_list


def _normalize_joints(joints):
    joints -= joints[0]  # Root at the origin
    # Scale joint positions such that the first thumb link length is equal to the shadow hand's link
    palm_vec1 = np.mean(np.vstack((joints[5], joints[9])), axis=0)
    palm_vec2 = np.mean(np.vstack((joints[13], joints[17])), axis=0)
    palm_normal = np.cross(palm_vec1, palm_vec2)
    palm_normal /= np.linalg.norm(palm_normal)
    n_R_j = _plane_normalization(palm_normal)
    joints = joints @ n_R_j.T
    palm_vec = np.mean(np.vstack((joints[5], joints[9], joints[13], joints[17])), axis=0)
    palm_vec /= np.linalg.norm(palm_vec)
    zrot = np.arccos(np.clip(np.dot(np.array([1, 0, 0]), palm_vec), -1.0, 1.0))
    w_R_n = zrot_matrix(zrot).numpy()[:3, :3]
    joints = joints @ w_R_n
    normalizer = 0.9972462 / np.linalg.norm(joints[9]-joints[0])
    joints *= normalizer
    return joints


def _plane_normalization(normal):
    ct = - normal[2] / np.linalg.norm(normal)  # Cosine of theta
    st = np.sqrt(np.sum(normal[:2]**2) / np.sum(normal**2))  # Sine of theta
    u1 = -normal[1] / np.linalg.norm(normal)
    u2 = normal[0] / np.linalg.norm(normal)
    rot = np.array([[ct + u1**2*(1-ct), u1*u2*(1-ct)     , u2*st ],
                    [u1*u2*(1-ct)     , ct + u2**2*(1-ct), -u1*st],
                    [-u2*st           , u1*st            , ct    ]])
    return rot