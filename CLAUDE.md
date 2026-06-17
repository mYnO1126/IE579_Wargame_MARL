# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Agent-based war game simulation of the Battle of the Valley of Tears (Yom Kippur War, 1973),
built for KAIST IE571. Python runs the discrete-time combat simulation over real Golan Heights
terrain extracted from QGIS; results (CSVs + frames) are exported for visualization in Unity.

### Status & pointers (2026-06-17)
- **Repo**: https://github.com/mYnO1126/IE579_Wargame_MARL (public). Local folder may be named
  `IE571_2025` or `IE579_Wargame_MARL` — git is unaffected by the folder name.
- **Run/test via the `wargame` conda env**: `conda run -n wargame python main.py ...`
  (env has numpy 1.26.4 / matplotlib 3.11, despite the `numpy==1.23.0` pin in requirements.txt).
- **`NOTES.txt`** = running work log + known issues/caveats; its content is the source for commit
  messages (write it in Korean, translate to English when committing). **Do NOT add a
  `Co-Authored-By: Claude` trailer to commits** — commits should be authored by the user only.
- **In progress: MARL.** Each troop is to become an RL agent. Design spec lives in
  `rl/DESIGN.md`; next step is a PettingZoo environment skeleton under `rl/`.

## Commands

```bash
# Environment (Python >= 3.12; project developed on 3.13)
conda create -n wargame python=3.13
conda activate wargame
pip install -r requirments.txt      # NOTE: file is misspelled "requirments.txt"; README references "requirements.txt"

# Run the simulation
python main.py
python main.py --plot True          # plot team strength at end of run (default: True)
python main.py --save_frames True   # save a frame every sim-minute (slow; default: False)
python main.py --save_tactics True  # save a tactical frame every 10 sim-minutes (default: True)
```

There is no test suite, linter, or build step. `modules/unit_definitions.py` has a `__main__`
block that exercises the ballistics interpolation helpers (`python -m modules.unit_definitions`).
Map preprocessing from raw GeoTIFF/shapefiles lives in `map/preprocess.ipynb` and produces the
bundled `map/golan_full_dataset_cropped.npz`; it is not part of the normal run.

## Output

Each run writes to a fresh `res/res<N>/` directory (auto-incremented by `utils.initialize_folders`):
`battle_log.csv` (every shot), `status_data.csv` (per-troop status over time), `plot.png`, and
`frames/` + `frames_tactics/` images. `res/` is gitignored.

## Architecture

The simulation is a single fixed-step loop in `main.py:main`. Each iteration advances
`current_time` by `TIME_STEP` (1.0 minute, defined in `modules/map.py`) up to `MAX_TIME`
(2880 min = 2 days). RNG is seeded (42) for reproducibility. The loop, per step:

1. Fire any due `TIMELINE` event (`handle_event`) — flips troop `active` / `can_move` flags.
2. Move troops (`update_troop_location_improved`).
3. Update mutual observation (`troop_list.update_observation`).
4. Assign targets to idle units, then fire any troop whose `next_fire_time` has arrived.
5. Periodically record status and render frames.

The five concepts that require reading multiple files to understand:

- **Troop & TroopList** (`modules/troop.py`) — `Troop` is one unit instance; its combat stats are
  pulled from a shared `UnitSpec`. `TroopList` owns all troops plus the `blue_troops`/`red_troops`
  and `blue_observed`/`red_observed` partitions, and drives targeting (`assign_targets`),
  observation, and firing (`fire`) across the whole force. Targeting only ever considers the
  *observed* enemy list, not all enemies.

- **Unit data tables** (`modules/unit_definitions.py`) — the single source of truth for unit
  capabilities. `UNIT_SPECS` maps each weapon name to a `UnitSpec` carrying range, speed, and
  *functions* (closures) for hit probability (`ph_func`), kill outcome (`pk_func`), damage, target
  acquisition delay, and reload time. Direct fire rolls Ph then Pk → `HitState`; indirect fire
  instead computes a shell landing point and lethal radius via the `curved_traj_weapon_data`
  ballistics tables. Also holds `AMMUNITION_DATABASE` / `SUPPLY_DATABASE` (currently unused — see
  Conventions). Tanks have multi-state damage (mobility/firepower/catastrophic kill); other units
  are killed outright.

- **Phased order of battle** (`modules/placement.py` + `modules/timeline.py`) — `PLACEMENT`
  declares each formation's spawn box (`loc`), optional destination box (`dest`), composition
  (`comp`), and a `phase` tag (e.g. `P1-1`, `P2-1`, `P3`, `P4`). At startup `main.py` samples
  concrete spawn coordinates and goal points into these dicts (`grid_sample_no_overlap`) and builds
  `Troop` objects. `TIMELINE` is the script: each `TimelineEvent` lists `(team, phase)` pairs in
  `active_on` / `move_on` that turn formations on and release them to move at a given sim-time.
  Team + phase is the join key between placement and timeline.

- **Terrain & movement** (`modules/map.py`) — `Map` loads raster layers (DEM, slope, aspect, and
  road/lake/wood/stream masks) from the `.npz`. It builds a per-cell movement `cost_map` (slope and
  terrain-type weighted; lakes/steep slopes are `inf` = impassable) and provides `is_visible` for
  line-of-sight checks against the DEM. Pathfinding offers A* (`astar_pathfinding`) and Dijkstra
  flow fields (`build_flow_field`, cached per goal on the map); tanks/APCs use flow fields, others
  use A*. `TacticalManager` computes role-specific destinations (flanking, ambush, cover, fire
  support). A troop's actual per-step velocity comes from `Troop.compute_velocity_advanced`, which
  follows its cached path and falls back to direct movement with obstacle avoidance.

- **History/output** (`modules/history.py`) — accumulates the battle log, per-troop status series,
  and visualization coordinates, and owns all matplotlib rendering (`draw_troop_positions`,
  `create_tactical_overview`, `plot_team_strength_over_time`). Note visualization coordinates are
  remapped for Unity: x→x, **z (elevation, minus reference altitude)→y**, and **(height − y)→z**.

### Coordinate & unit conventions

Grid coordinates are in pixels at 10 m/pixel (`Map.resolution_m`); `get_distance` multiplies pixel
distance by 0.01 to get km, and weapon `range_km` is compared in km. Time is in minutes; in-game
clock starts at 13:55 (`int((13*60+55+current_time)//60) % 24`), and night (19:00–06:00) slows
movement and degrades hit/acquisition.

### Faction asymmetry

BLUE (Israeli) units get buffs applied in `troop.py`: `BLUE_RANGE_BUFF` (range ×1.2),
`BLUE_OBS_BUFF` (observation ×1.2), and `BLUE_HIT_PROB_BUFF` (enemy fire *against* blue ×0.8).

## Conventions

- **Never delete code — comment it out instead.** This codebase already follows that practice
  heavily: large blocks of alternative implementations, the ammo/supply/resupply system, and
  earlier `PLACEMENT` layouts are all retained as comments. Preserve them.
- **Mark code that Claude comments out** with a `#!CLAUDE` tag on the line(s) immediately above
  the disabled code (mirroring the existing `#!TEMP` marker style), so it's clear which commented
  blocks were disabled by Claude versus pre-existing ones. Briefly note why.
- `#!TEMP >>>>` / `#!TEMP <<<<` markers bracket the newer pathfinding/observation/targeting code
  that superseded older inline logic. Treat the marked block as the active implementation.
- Inline comments and many `print` debug statements are in Korean; keep new comments consistent
  with the surrounding style.
- Prioritize readability when changing code.
