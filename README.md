# IE579 Wargame MARL
### KAIST IE579 2026 Spring 

## War Game

### Scenario
Valley of Tears, Yom Kippur War, 1973 Arab-Israeli War

### Simulation Structure

Terrain data extracted from QGIS

Simulation implemented with Python

Visualization implemented with Unity

### Installation 

Clone repo and install [requirements.txt](git@github.com:mYnO1126/IE579_Wargame_MARL.git) in a
[**Python>=3.12**](https://www.python.org/) environment.

Python 3.13.3 used for simulation.

Follow the steps below in order to reproduce the results.


1. Clone repo

```bash
git clone https://github.com/mYnO1126/IE579_Wargame_MARL.git  # clone
cd IE579_Wargame_MARL
```

2.  Make a Conda Environment
    <details>
        <summary>Install Conda if necessary</summary>
        Install Conda that fits with your machine

        ```bash
        wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
        bash Anaconda3-2024.02-1-Linux-x86_64.sh
        conda init
        ```

    </details>

```bash
conda create -n wargame python=3.13
conda activate wargame
```

3. Install Requirements

```bash
pip install -r requirements.txt  # install
```

### Run War Game Simulation

```bash
python main.py

# python main.py --plot True    # show plot of team strength after simulation is done, default: True
# python main.py --save_frames True # save frames every minute during simulation (slow), default: False
# python main.py --save_tactics True # save tactic frames every 10 minutes, default: True

```

results saved in res/res*

```bash
├── res/
│   ├── res*/
│   │   ├── frames/
│   │   │   └── frame_*.png
│   │   ├── frames_tactics/
│   │   │   └── tactical_*.png
│   │   ├── battle_log.csv
│   │   ├── plot.png
│   │   ├── status_data.csv
│   │   └── visualization_data.csv

```

### MARL (Reinforcement Learning)

Each troop is treated as an RL agent. The RL code (`rl/`) wraps the existing simulation as a
PettingZoo-style parallel multi-agent environment and reuses its dynamics (firing Ph/Pk, line of
sight, terrain, observation); only the per-unit decisions (move / target / engage) are learned.
Design spec: [`rl/DESIGN.md`](rl/DESIGN.md).

The RL code needs **PyTorch** in the `wargame` env (the base simulation only needs
numpy/pandas/matplotlib):

```bash
conda activate wargame
pip install torch        # use the CUDA build if you have a GPU
```

All commands run from the repo root.

**1. Sanity-check the environment (random policy, no learning):**

```bash
python -m rl.random_rollout            # run one episode with random actions
```

**2. Train a policy** (parameter-shared PPO; one team learns, the other plays a scripted
rule-based policy). MAPPO centralized critic is built in.

```bash
python -m rl.train --team red  --iters 120 --workers 8
python -m rl.train --team blue --iters 80  --workers 8
```

- `--team {red,blue}` : which side the policy controls.
- `--workers N` : number of parallel rollout processes (`1` = single env; more ≈ N× faster).
- other flags: `--iters --roll --lr --seed --cuda` (see `python -m rl.train -h`).
- the trained policy is saved to `rl/policies/ippo_<team>.pt`.

**3. Visualize an episode** (saves PNG frames to `rl/viz/ep_*/`):

```bash
python -m rl.visualize_episode --mode board --seed 3
# --mode {board,tactical}   --every N (sim-min per frame)   --legend   --policy {scripted,random}
```

**4. Evaluate a trained policy vs the rule-based baseline** (the project's success criterion is
*improvement over rule-based*, not absolute win rate):

```bash
python -m rl.evaluate --ckpt rl/policies/ippo_red.pt --episodes 200
```

Prints a table comparing the MARL policy and the rule-based (scripted) baseline
(win / loss / survival). Outputs under `rl/policies/` and `rl/viz/` are gitignored.

