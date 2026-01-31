import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from quantum_control_rl_server.remote_env_tools import Client
from trapped_ion_cat_sim_function import (
    trapped_ion_cat_sim,
    characteristic_grid,
    prepare_characteristic_distribution,
)

logger = logging.getLogger("RL")
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

client_socket = Client()
(host, port) = ("127.0.0.1", 5555)
client_socket.connect((host, port))

N_STEPS = 120
N_SEGMENTS = 60
SEG_LEN = N_STEPS // N_SEGMENTS
T_STEP = 10.0
SAMPLE_EXTENT = 3.0
N_BOSON = 30
ALPHA_CAT = 2.0

TRAIN_POINTS_STAGE1 = 80
TRAIN_POINTS_STAGE2 = 160
TRAIN_POINTS_STAGE3 = 240
TRAIN_STAGE1_EPOCHS = 200
TRAIN_STAGE2_EPOCHS = 400

SMOOTH_LAMBDA = 0.0
SMOOTH_PHI_WEIGHT = 1.0
SMOOTH_AMP_WEIGHT = 0.2
REWARD_SCALE = 30.0
REWARD_CLIP = None

N_SHOTS_TRAIN = 0
N_SHOTS_EVAL = 0

CHAR_UNIFORM_MIX = 0.1
CHAR_POINTS, CHAR_TARGET, CHAR_WEIGHTS, CHAR_AREA = prepare_characteristic_distribution(
    alpha_cat=ALPHA_CAT,
    n_boson=N_BOSON,
    extent=SAMPLE_EXTENT,
    grid_size=41,
    cat_parity="even",
    mix_uniform=CHAR_UNIFORM_MIX,
)
FINAL_POINTS, FINAL_TARGET, FINAL_WEIGHTS, FINAL_AREA = prepare_characteristic_distribution(
    alpha_cat=ALPHA_CAT,
    n_boson=N_BOSON,
    extent=SAMPLE_EXTENT,
    grid_size=61,
    cat_parity="even",
    mix_uniform=CHAR_UNIFORM_MIX,
)


def _smoothness_penalty(phi_r, phi_b, amp_r, amp_b):
    dphi_r = np.diff(phi_r)
    dphi_b = np.diff(phi_b)
    damp_r = np.diff(amp_r)
    damp_b = np.diff(amp_b)
    phi_pen = 0.5 * (np.mean(dphi_r ** 2) + np.mean(dphi_b ** 2))
    amp_pen = 0.5 * (np.mean(damp_r ** 2) + np.mean(damp_b ** 2))
    return SMOOTH_PHI_WEIGHT * phi_pen + SMOOTH_AMP_WEIGHT * amp_pen


def _sample_characteristic_points(rng, n_points):
    idx = rng.choice(len(CHAR_POINTS), size=n_points, replace=True, p=CHAR_WEIGHTS)
    points = [CHAR_POINTS[i] for i in idx]
    targets = CHAR_TARGET[idx]
    weights = CHAR_WEIGHTS[idx]
    return points, targets, weights


def _select_train_points(epoch, rng):
    if epoch < TRAIN_STAGE1_EPOCHS:
        return _sample_characteristic_points(rng, TRAIN_POINTS_STAGE1)
    if epoch < TRAIN_STAGE2_EPOCHS:
        return _sample_characteristic_points(rng, TRAIN_POINTS_STAGE2)
    return _sample_characteristic_points(rng, TRAIN_POINTS_STAGE3)

done = False
eval_log_path = os.path.join(os.getcwd(), "eval_fidelity.csv")
while not done:
    message, done = client_socket.recv_data()
    logger.info("Received message from RL agent server.")
    logger.info("Time stamp: %f", time.time())

    if done:
        logger.info("Training finished.")
        break

    epoch_type = message["epoch_type"]

    if epoch_type == "final":
        logger.info("Final Epoch")
        locs = message["locs"]
        scales = message["scales"]
        for key in locs.keys():
            logger.info("locs[%s]:", key)
            logger.info(locs[key][0])
            logger.info("scales[%s]:", key)
            logger.info(scales[key][0])
        phi_r_final = np.repeat(np.array(locs["phi_r"][0]), SEG_LEN)
        phi_b_final = np.repeat(np.array(locs["phi_b"][0]), SEG_LEN)
        amp_r_final = np.repeat(np.array(locs["amp_r"][0]), SEG_LEN)
        amp_b_final = np.repeat(np.array(locs["amp_b"][0]), SEG_LEN)

        _, final_fidelity, rho_final, rho_target = trapped_ion_cat_sim(
            phi_r_final,
            phi_b_final,
            amp_r=amp_r_final,
            amp_b=amp_b_final,
            n_boson=N_BOSON,
            omega=2 * np.pi * 0.002,
            t_step=T_STEP,
            alpha_cat=ALPHA_CAT,
            cat_parity="even",
            sample_points=FINAL_POINTS,
            target_values=FINAL_TARGET,
            sample_weights=FINAL_WEIGHTS,
            sample_area=FINAL_AREA,
            reward_scale=REWARD_SCALE,
            reward_clip=REWARD_CLIP,
            n_shots=0,
            return_details=True,
            reward_mode="characteristic",
        )

        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        fidelity_path = os.path.join(output_dir, "final_fidelity.txt")
        with open(fidelity_path, "w", encoding="utf-8") as f:
            f.write(f"{final_fidelity:.6f}\n")
        logger.info("Final fidelity %.6f", final_fidelity)
        logger.info("Saved final fidelity to %s", fidelity_path)

        grid = np.linspace(-SAMPLE_EXTENT, SAMPLE_EXTENT, 121)
        chi_target = characteristic_grid(rho_target, grid, grid)
        chi_final = characteristic_grid(rho_final, grid, grid)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(
            chi_target,
            extent=[-SAMPLE_EXTENT, SAMPLE_EXTENT, -SAMPLE_EXTENT, SAMPLE_EXTENT],
            origin="lower",
            cmap="RdBu_r",
        )
        axes[0].set_title("Target cat characteristic")
        axes[0].set_xlabel("Re(alpha)")
        axes[0].set_ylabel("Im(alpha)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(
            chi_final,
            extent=[-SAMPLE_EXTENT, SAMPLE_EXTENT, -SAMPLE_EXTENT, SAMPLE_EXTENT],
            origin="lower",
            cmap="RdBu_r",
        )
        axes[1].set_title("Final state characteristic")
        axes[1].set_xlabel("Re(alpha)")
        axes[1].set_ylabel("Im(alpha)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        char_path = os.path.join(output_dir, "char_target_vs_final.png")
        fig.savefig(char_path, dpi=150)
        plt.close(fig)
        logger.info("Saved characteristic plot to %s", char_path)

        done = True
        logger.info("Training finished.")
        break

    action_batch = message["action_batch"]
    batch_size = message["batch_size"]
    epoch = message["epoch"]

    phi_r_coeff = action_batch["phi_r"].reshape([batch_size, -1])
    phi_b_coeff = action_batch["phi_b"].reshape([batch_size, -1])
    amp_r_coeff = action_batch["amp_r"].reshape([batch_size, -1])
    amp_b_coeff = action_batch["amp_b"].reshape([batch_size, -1])
    phi_r = np.repeat(phi_r_coeff, SEG_LEN, axis=1)
    phi_b = np.repeat(phi_b_coeff, SEG_LEN, axis=1)
    amp_r = np.repeat(amp_r_coeff, SEG_LEN, axis=1)
    amp_b = np.repeat(amp_b_coeff, SEG_LEN, axis=1)

    logger.info("Start %s epoch %d", epoch_type, epoch)

    rng = np.random.default_rng(epoch)
    if epoch_type == "evaluation":
        n_shots = N_SHOTS_EVAL
        sample_points, target_values, sample_weights = _sample_characteristic_points(
            rng, TRAIN_POINTS_STAGE3
        )
    else:
        n_shots = N_SHOTS_TRAIN
        sample_points, target_values, sample_weights = _select_train_points(epoch, rng)

    reward_data = np.zeros((batch_size))
    fidelity_data = None
    if epoch_type == "evaluation":
        fidelity_data = np.zeros((batch_size))
    for ii in range(batch_size):
        if epoch_type == "evaluation":
            reward, fidelity, _, _ = trapped_ion_cat_sim(
                phi_r[ii],
                phi_b[ii],
                amp_r=amp_r[ii],
                amp_b=amp_b[ii],
                n_boson=N_BOSON,
                omega=2 * np.pi * 0.002,
                t_step=T_STEP,
                alpha_cat=ALPHA_CAT,
                cat_parity="even",
                sample_points=sample_points,
                target_values=target_values,
                sample_weights=sample_weights,
                sample_area=CHAR_AREA,
                reward_scale=REWARD_SCALE,
                reward_clip=REWARD_CLIP,
                n_shots=n_shots,
                return_details=True,
                reward_mode="characteristic",
            )
            reward_data[ii] = reward
            fidelity_data[ii] = fidelity
        else:
            reward_data[ii] = trapped_ion_cat_sim(
                phi_r[ii],
                phi_b[ii],
                amp_r=amp_r[ii],
                amp_b=amp_b[ii],
                n_boson=N_BOSON,
                omega=2 * np.pi * 0.002,
                t_step=T_STEP,
                alpha_cat=ALPHA_CAT,
                cat_parity="even",
                sample_points=sample_points,
                target_values=target_values,
                sample_weights=sample_weights,
                sample_area=CHAR_AREA,
                reward_scale=REWARD_SCALE,
                reward_clip=REWARD_CLIP,
                n_shots=n_shots,
                reward_mode="characteristic",
            )
            smooth_pen = _smoothness_penalty(
                phi_r[ii], phi_b[ii], amp_r[ii], amp_b[ii]
            )
            reward_data[ii] -= SMOOTH_LAMBDA * smooth_pen

    R = np.mean(reward_data)
    std_R = np.std(reward_data)
    logger.info("Average reward %.3f", R)
    logger.info("STDev reward %.3f", std_R)
    if fidelity_data is not None:
        mean_fidelity = float(np.mean(fidelity_data))
        std_fidelity = float(np.std(fidelity_data))
        logger.info("Eval fidelity %.6f", mean_fidelity)
        write_header = not os.path.exists(eval_log_path)
        with open(eval_log_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("epoch,mean_fidelity,std_fidelity\n")
            f.write(f"{epoch},{mean_fidelity:.6f},{std_fidelity:.6f}\n")

    logger.info("Sending message to RL agent server.")
    logger.info("Time stamp: %f", time.time())
    client_socket.send_data(reward_data)
