from pathlib import Path
import logging
import multiprocessing as mp

import torch
import json
import numpy as np
import nlopt
from tqdm import tqdm

from sh_kinematics import JOINT_LIMITS, shadow_hand_fk
from tf import zrot_matrix
from visualization import visualize_frames, visualize_joints, visualize_joints_and_frames
from utils import load_dataset, _normalize_joints


FRAME_IDX = [1, 2, 4, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 24, 26, 27, 28]


def load_joints():
    data_path = Path(__file__).parents[1] / "ContactPose" / "data" / "contactpose_data" / "full28_use" / "bowl" / "annotations.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    
    joints = np.array(data["hands"][1]["joints"])
    joints = _normalize_joints(joints)
    return joints


def distance(frames, positions):
    frames = torch.vstack([frames[idx][0:3,3] for idx in FRAME_IDX])
    assert frames.shape == positions.shape
    return torch.sum(torch.norm(frames - positions, dim=0))


def ik_objective(x, grad, positions):
    x = torch.as_tensor(x.copy()).requires_grad_()
    frames = shadow_hand_fk(x)
    loss = distance(frames, positions)
    if grad.size > 0:
        grad[:] = torch.autograd.grad(loss, x)[0]
    return loss.item()


def mp_task(args):
    joints, jid = args
    joints = torch.as_tensor(joints)
    limit_low = np.array([0, 0, 0] + JOINT_LIMITS["lower"])
    limit_high = np.array([2*np.pi, 2*np.pi, 2*np.pi] + JOINT_LIMITS["upper"])
    opt = nlopt.opt(nlopt.LD_MMA, 23)
    opt.set_lower_bounds(limit_low)
    opt.set_upper_bounds(limit_high)
    opt.set_min_objective(lambda x, grad: ik_objective(x, grad, joints))
    opt.set_xtol_rel(1e-4)
    lsol = np.inf
    xsol = None
    for _ in range(1):
        theta_start = np.random.uniform(size=23) * (limit_high - limit_low) + limit_low
        x = opt.optimize(theta_start)
        loss = opt.last_optimum_value()
        if loss < lsol:
            xsol = x
            lsol = loss
    return (xsol.tolist(), lsol, jid)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    frames = shadow_hand_fk(torch.zeros(23, dtype=torch.float64))
    frames = [frames[idx] for idx in FRAME_IDX]
    data = load_dataset(Path(__file__).parents[1] / "ContactPose")
    
    nprocesses = mp.cpu_count() - 1
    pool = mp.Pool(nprocesses)
    results = []
    for result in tqdm(pool.imap_unordered(mp_task, zip(data, range(len(data)))), total=len(data)):
        results.append(result)
    
    with open(Path(__file__).parent / "sh_joints.json", "w") as f:
        json.dump(results, f)
