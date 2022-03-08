from pathlib import Path
import logging
import multiprocessing as mp
from datetime import datetime

import torch
import json
import numpy as np
import nlopt
from tqdm import tqdm

from sh_kinematics import JOINT_LIMITS, shadow_hand_fk

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
    positions, jid = args
    positions = torch.as_tensor(positions)
    limit_low = np.concatenate((np.zeros(3), JOINT_LIMITS["lower"]))
    limit_high = np.concatenate((2*np.pi*np.ones(3), JOINT_LIMITS["upper"]))
    opt = nlopt.opt(nlopt.LD_MMA, 23)
    opt.set_lower_bounds(limit_low)
    opt.set_upper_bounds(limit_high)
    opt.set_min_objective(lambda x, grad: ik_objective(x, grad, positions))
    opt.set_xtol_rel(1e-4)
    lsol = np.inf
    xsol = None
    for _ in range(100):
        theta_start = np.random.uniform(size=23) * (limit_high - limit_low) + limit_low
        x = opt.optimize(theta_start)
        loss = opt.last_optimum_value()
        if loss < lsol:
            xsol = x
            lsol = loss
    return (xsol.tolist(), lsol, jid)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Create the pool before loading the datasets since each worker gets a copy of the memory
    nprocesses = mp.cpu_count() - 1
    pool = mp.Pool(nprocesses)

    frames = shadow_hand_fk(torch.zeros(23, dtype=torch.float64))
    frames = [frames[idx] for idx in FRAME_IDX]
    data = load_dataset(Path(__file__).parents[1] / "ContactPose")
    
    results = []
    for result in tqdm(pool.imap_unordered(mp_task, zip(data, range(len(data)))), total=len(data),
                       desc="Task progress"):
        results.append(result)
    
    path = (Path(__file__).parent / "saves")
    path.mkdir(exist_ok=True)
    with open(path / "sh_joints.json", "w") as f:
        json.dump(results, f)
    path = (Path(__file__).parent / "backup")
    path.mkdir(exist_ok=True)
    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    with open(path / ("sh_joints"+date+".json"), "w") as f:
        json.dump(results, f)