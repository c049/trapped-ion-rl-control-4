# Trapped-Ion RL Control

This repository provides a reinforcement-learning control server with a TCP/IP client interface for quantum-control experiments and simulations. It includes multiple example environments (pi-pulse, OCT-style pi-pulse, and trapped-ion cat-state preparation), along with plotting and analysis scripts.

## Project layout
- `quantum_control_rl_server/`: core RL server code, PPO training, environment wrappers, and TCP/IP utilities.
- `examples/pi_pulse/`: baseline pi-pulse example (training server + client + analysis).
- `examples/pi_pulse_oct_style/`: OCT-style pi-pulse variant (training server + client + analysis).
- `examples/trapped_ion_cat/`: trapped-ion cat-state environment, training server, client, and plotting.
- `examples/trapped_ion_cat/outputs/`: saved plots and artifacts (included in this repo).

## Requirements
The project was tested with the conda environment defined in `qcrl-server-tf240.yml` (TensorFlow 2.4.0 and TF-Agents 0.6.0). CPU-only setups work, but training may be slower.

## Installation
Create and activate the conda environment:
```sh
conda env create -f qcrl-server-tf240.yml
conda activate qcrl-server
```
Install the package in editable mode:
```sh
pip install -e .
```

## Running the examples
Open two terminals, activate the environment in both, and run the training server and client in the same example directory.

Pi-pulse:
```sh
cd examples/pi_pulse
python pi_pulse_training_server.py
```
In another terminal:
```sh
cd examples/pi_pulse
python pi_pulse_client.py
```

OCT-style pi-pulse:
```sh
cd examples/pi_pulse_oct_style
python pi_pulse_oct_style_training_server.py
```
In another terminal:
```sh
cd examples/pi_pulse_oct_style
python pi_pulse_oct_style_client.py
```

Trapped-ion cat-state:
```sh
cd examples/trapped_ion_cat
python trapped_ion_cat_training_server.py
```
In another terminal:
```sh
cd examples/trapped_ion_cat
python trapped_ion_cat_client.py
```

## Outputs
Plots and evaluation artifacts are saved under `examples/trapped_ion_cat/outputs/` (for example: training curves, pulse sequences, fidelity curves, and Wigner function comparisons). These files are tracked in this repository.

## License
See `LICENSE`.
