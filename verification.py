import logging
from pathlib import Path
import argparse
from datetime import datetime

import json
import numpy as np

from utils import load_dataset
from visualization import visualize_comparison, visualize_joints
from sh_kinematics import shadow_hand_fk


logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        help="Selects the verification task",
                        choices=["hands", "solutions"],
                        default="solutions")
    parser.add_argument("--loglvl",
                        help="Logger levels",
                        choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        default="INFO")
    args = parser.parse_args()
    return args


def verify_hands():
    data = load_dataset(Path(__file__).parents[1] / "ContactPose")
    results = {}
    logger.info(f"Starting hand verification for {len(data)} samples")
    ax = None
    for idx, joints in enumerate(data):
        try:
            ax = visualize_joints(joints, ax)
            x = input("Press enter to verify, anything else to flag as faulty:")
            print(f"Verified: {idx}" if x == "" else f"Faulty: {idx}")
            results[idx] = (x == "")
        except Exception as e:
            logger.warning(e)
            pass
    with open(Path(__file__).parent / "saves" / "hand_verification.json", "w") as f:
        json.dump(results, f)
    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    with open(Path(__file__).parent / "backup" / ("hand_verification_"+date+".json"), "w") as f:
        json.dump(results, f)
    logger.info("Hand verification saved")


def verify_solutions():
    path = Path(__file__).parent / "saves" / "sh_joints.json"
    try:
        with open(path, "r") as f:
            joint_df = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError("Shadow hand joint configurations are missing. Please make sure to"
                                f" run the IK optimization first.\n Original Error: {e}")
    joint_df = sorted(joint_df, key=lambda x: x[2])

    data = load_dataset(Path(__file__).parents[1] / "ContactPose")
    assert len(joint_df) == len(data), "Joint data and ContactPose dataset sizes are inconsistent"
    
    logger.info(f"Starting solution verification for {len(data)} samples")
    ax = None
    results = {}
    for idx in range(len(joint_df)):
        ax = visualize_comparison(data[idx], shadow_hand_fk(np.array(joint_df[idx][0])), ax, False)
        x = input("Press enter to verify, anything else to flag as faulty:")
        print(f"Verified: {idx}" if x == "" else f"Faulty: {idx}")
        results[idx] = (x == "")
    with open(Path(__file__).parent / "saves" / "joint_verification.json", "w") as f:
        json.dump(results, f)
    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    with open(Path(__file__).parent / "backup" / ("joint_verification_"+date+".json"), "w") as f:
        json.dump(results, f)
    logger.info("Joint verification saved")

if __name__ == "__main__":
    args = parse_args()
    loglvls = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "ERROR": logging.ERROR
    }
    logging.basicConfig(level=loglvls[args.loglvl])
    (Path(__file__).parent / "saves").mkdir(exist_ok=True)
    (Path(__file__).parent / "backup").mkdir(exist_ok=True)
    if args.task == "hands":
        verify_hands()
    elif args.task == "solutions":
        verify_solutions()
