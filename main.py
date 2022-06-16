from __future__ import annotations
import numpy as np
import os
import matplotlib.backend_bases

if os.name == "darwin":
    matplotlib.use("MacOSX")  # for mac
else:
    matplotlib.use("TkAgg")  # for unix/windows

import matplotlib.pyplot as plt

# Size of the figure on the plot
plt.rcParams["figure.figsize"] = (15, 15)
plt.ion()  # interactive mode
plt.style.use("dark_background")

target_point = np.array([-3.0, 0])
anchor_point = np.array([0, 0])

is_running = True


def button_press_event(event):
    global target_point
    target_point = np.array([event.xdata, event.ydata])


def press(event):
    global is_running
    print('press', event.key)
    if event.key == "escape":
        is_running = False


def on_close(event):
    global is_running
    is_running = False


fig, _ = plt.subplots()
fig.canvas.mpl_connect("close_event", on_close)
fig.canvas.mpl_connect("button_press_event", button_press_event)
fig.canvas.mpl_connect("key_press_event", press)

length_joint = 2.0
theta_1 = np.deg2rad(-10)
theta_2 = np.deg2rad(-30)
theta_3 = np.deg2rad(40)


def rotation(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return R


def d_rotation(theta):
    dR = np.array([
        [-np.sin(theta), -np.cos(theta)],
        [np.cos(theta), -np.sin(theta)]
    ])
    return dR

alpha = 1e-2

while is_running:
    plt.clf()
    plt.title(f'theta_1: {round(np.rad2deg(theta_1))} theta_2: {round(np.rad2deg(theta_2))}')
    t = np.array([0.0, 1.0]) * length_joint

    R1 = rotation(theta_1)
    dR1 = d_rotation(theta_1)
    R2 = rotation(theta_2)
    dR2 = d_rotation(theta_2)
    R3 = rotation(theta_3)
    dR3 = d_rotation(theta_3)

    joints = []
    joints.append(anchor_point)

    point_1 = np.dot(R1, t)
    joints.append(point_1)

    point_2 = np.dot(R1, t) + np.dot(R1, np.dot(R2, t))
    joints.append(point_2)

    point_3 = np.dot(R1, t) +np.dot(R1, np.dot(R2, t)) +  np.dot(R1, np.dot(R2, np.dot(R3, t)))
    joints.append(point_3)


    # print(joints)
    np_joints = np.array(joints)


    d_theta_1 = np.sum(2 * (point_3 - target_point) * (dR1 @ t))
    theta_1 -= d_theta_1 * alpha

    d_theta_2 = np.sum(2 * (point_3 - target_point) * (R1 @ dR2 @ t))
    theta_2 -= d_theta_2 * alpha

    d_theta_3 = np.sum(2 * (point_3 - target_point) * (R1 @ R2 @ dR3 @ t))
    theta_3 -= d_theta_3 * alpha


    loss = np.sum((target_point - point_3) ** 2)/len(point_3)


    if len(np_joints):
        plt.plot(np_joints[:, 0], np_joints[:, 1])
    plt.scatter(target_point[0], target_point[1], s=50, c='r')

    plt.title(f'loss:{loss}')
    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    plt.draw()
    plt.pause(1e-1)
input("end")