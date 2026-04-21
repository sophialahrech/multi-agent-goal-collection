# Multi-agent Goal Collection — PDM4AR, ETH Zürich 2025

> Decentralized planning and control for a fleet of differential-drive robots navigating a warehouse to collect and deliver goals.  
> Group project — PDM4AR Final Exercise, ETH Zürich, Fall 2025.

---

## Problem overview

A fleet of robots must autonomously navigate a warehouse environment, collect goals scattered across the map, and deliver them to designated collection points. Each robot carries **at most one goal at a time** and operates with only **local sensor information** after an initial global planning phase.

The agent runs in **closed-loop** at 10 Hz: at each timestep it receives local observations from a simulated 360° Lidar (5m range) and must return wheel velocity commands.

![Environment](https://pdm4ar.github.io/exercises/img/multiagent_collection_example.png)

---

## System architecture

The solution is split into two phases:

### 1. One-time global planning (before simulation starts)

`Pdm4arGlobalPlanner.send_plan()` receives full initial knowledge of the environment:
- Initial positions of all robots
- Locations of all goals and collection points
- Full map (boundaries + static obstacles)

We use this phase for:
- **Task allocation** — assign goals to robots to minimise total travel distance
- **Path pre-computation** — compute collision-free waypoint sequences per robot
- **Coordination** — design strategies to avoid deadlocks at collection points

The global plan is serialized and broadcast to all agents before the simulation starts.

### 2. Decentralized agent control (closed-loop at 10 Hz)

Each robot runs the same `Pdm4arAgent` independently, with no communication:

```
Global plan (received once)
        +
Local Lidar obs (5m range, 360°, line-of-sight only)
        │
        ▼
  High-level replanning (every ~0.5s)
  → Check if assigned goal still available
  → Replan if goal taken by another robot
        │
        ▼
  Local collision avoidance (every 0.1s)
  → Detect dynamic obstacles (other robots)
  → Adjust commands in real time
        │
        ▼
  DiffDriveCommands [omega_l, omega_r]
```

---

## Robot model

**State:** `[x, y, ψ]` — position and heading  
**Commands:** `[omega_l, omega_r]` — left/right wheel angular velocities (rad/s)

Differential drive kinematics from `dg-commons`:
```
vx = r/2 · (omega_r + omega_l) · cos(ψ)
vy = r/2 · (omega_r + omega_l) · sin(ψ)
ψ̇  = r/L · (omega_r - omega_l)
```

---

## Rules

- **No communication after global planning** — no shared memory, no global variables, no inter-agent messaging
- **Automatic pickup** — robot enters goal polygon → goal automatically collected
- **Automatic delivery** — robot enters collection point polygon → goal automatically deposited
- **Collision disables robot** — collision with obstacle or another robot → robot is disabled for the rest of the simulation
- **Partial observability** — only non-occluded robots and goals within 5m are visible

---

## Scoring

```
score  = num_goals_delivered × 100
score -= num_collided_players × 500
score += (max_sim_time - task_accomplishment_time) × 10
score -= total_travelled_distance × 0.5
score -= total_actuation_effort × 0.01
score -= max(0, avg_computation_time - 0.1) × 100
score -= max(0, global_planning_time - 60) × 10
```

**Priority order:** deliver goals → avoid collisions → finish fast → minimise distance

Reference scores from the TA baseline:

| Test case | Reference score |
|---|---|
| config_1 | 489.43 |
| config_2 | 904.30 |
| config_3 | 1255.45 |

---

## Code structure

```
src/pdm4ar/exercises/ex14/
├── agent.py          # GlobalPlanner + Agent implementation (send_plan, on_episode_init,
│                     #   on_receive_global_plan, get_commands)
└── batch_runner.py   # CLI script to run all configs and log scores to JSONL
```

---

## Stack

| Tool | Role |
|---|---|
| Python 3.11 | Core implementation |
| dg-commons | Robot models, simulator structures |
| Pydantic | Global plan serialization |
| NumPy / SciPy | Path planning, geometry |
| Shapely | Polygon intersection, obstacle checking |

---

## Run

```bash
python path/to/src/pdm4ar/main.py --exercise 14
```

---

**Course:** PDM4AR — Planning and Decision Making for Autonomous Robots, ETH Zürich, Fall 2025  
**Team:** les-gargouilles
