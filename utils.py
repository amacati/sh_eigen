import logging
from pathlib import Path

import numpy as np
import json
from tqdm import tqdm
from sh_kinematics import TOTAL_LENGTH

from tf import zrot_matrix



logger = logging.getLogger(__name__)

def load_dataset(path: Path):
    # Try to load the list of verified dataset samples
    verification_path = Path(__file__).parent / "saves" / "hand_verification.json"
    if verification_path.exists():
        with open(verification_path, "r") as f:
            verification_data = json.load(f)
        verified_indices = [int(idx) for idx, verified in verification_data.items() if verified]
        verified_indices = sorted(verified_indices)  # Make sure the samples are in correct order
    else:
        logger.warning("No dataset verification file found! Dataset will contain bad samples.")
        verified_indices = None
    # Load the dataset from cache if available
    cache_path = Path(__file__).parent / ".cache" / "dataset_cache.json"
    if cache_path.exists():
        logger.info("Loading data from cache")
        try:
            with open(cache_path, "r") as f:
                joint_list = json.load(f)
            if verified_indices:
                return [np.array(joint_list[idx]) for idx in verified_indices]
            return [np.array(joints) for joints in joint_list]
        except json.JSONDecodeError:
            logger.warning("Corrupted cache read, reloading dataset from source")
            pass
    data_path = Path(path) / "data" / "contactpose_data"
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
    if verified_indices:
        return [joint_list[idx] for idx in verified_indices]
    return joint_list


def _normalize_joints(joints):
    joints -= joints[0]  # Root at the origin
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
    # Scale joint positions such that the hand's total link length is equal to the shadow hand's
    total_len = 0
    for i in range(5):
        base = 1 + i*4
        total_len += np.linalg.norm(joints[0], joints[base])
        for j in range(3):
            total_len += np.linalg.norm(joints[base+j] - joints[base+j+1])
    normalizer = TOTAL_LENGTH / total_len
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