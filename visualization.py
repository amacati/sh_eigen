import matplotlib.pyplot as plt
import numpy as np


def visualize_joints(joints):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])
    for i in range(5):
        base = 1 + i*4
        ax.plot([joints[0, 0], joints[base, 0]], [joints[0, 1], joints[base, 1]], zs=[joints[0, 2], joints[base, 2]])
        for j in range(3):
            ax.plot([joints[base+j, 0], joints[base+j+1, 0]], [joints[base+j, 1], joints[base+j+1, 1]],zs=[joints[base+j, 2],joints[base+j+1, 2]])
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()


def visualize_frames(frames):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    minc = np.array([np.inf, np.inf, np.inf])  # Necessary for scaling 3D plot
    maxc = np.array([-np.inf, -np.inf, -np.inf])
    for frame in frames:
        pos = frame[0:3,3]
        minc = np.minimum(minc, pos)
        maxc = np.maximum(maxc, pos)
        ex, ey, ez = frame[0:3,0]*.1 + pos, frame[0:3,1]*.1 + pos, frame[0:3,2]*.1 + pos
        ax.scatter(*pos, color="0")
        for ei, color in zip([ex, ey, ez], ["r", "g", "b"]):
            minc = np.minimum(minc, ei)
            maxc = np.maximum(maxc, ei)
            ax.plot([pos[0], ei[0]], [pos[1], ei[1]], zs=[pos[2], ei[2]], color=color)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_box_aspect((maxc-minc))
    plt.show()


def visualize_joints_and_frames(joints, frames):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])
    minc = np.min(joints, axis=0)  # Necessary for scaling 3D plot
    maxc = np.max(joints, axis=0)
    for i in range(5):
        base = 1 + i*4
        ax.plot([joints[0, 0], joints[base, 0]], [joints[0, 1], joints[base, 1]], zs=[joints[0, 2], joints[base, 2]])
        for j in range(3):
            ax.plot([joints[base+j, 0], joints[base+j+1, 0]], [joints[base+j, 1], joints[base+j+1, 1]],zs=[joints[base+j, 2],joints[base+j+1, 2]])
    for frame in frames:
        pos = frame[0:3,3]
        minc = np.minimum(minc, pos)
        maxc = np.maximum(maxc, pos)
        ex, ey, ez = frame[0:3,0]*.1 + pos, frame[0:3,1]*.1 + pos, frame[0:3,2]*.1 + pos
        ax.scatter(*pos, color="0")
        for ei, color in zip([ex, ey, ez], ["r", "g", "b"]):
            minc = np.minimum(minc, ei)
            maxc = np.maximum(maxc, ei)
            ax.plot([pos[0], ei[0]], [pos[1], ei[1]], zs=[pos[2], ei[2]], color=color)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_box_aspect((maxc-minc))
    plt.show()