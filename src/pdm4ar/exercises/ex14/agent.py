import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from dg_commons import PlayerName
from dg_commons.sim import InitSimGlobalObservations, InitSimObservations, SimObservations
from dg_commons.sim.agents import Agent, GlobalPlanner
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands, DiffDriveState
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.models.obstacles import StaticObstacle
from pydantic import BaseModel
from shapely.geometry import LineString, LinearRing, Point, Polygon
from shapely.strtree import STRtree


DEBUG = False

class GoalSpec(BaseModel):
    goal_id: str
    center: Tuple[float, float]
    radius: float


class CollectionSpec(BaseModel):
    point_id: str
    center: Tuple[float, float]
    radius: float


class GridSpec(BaseModel):
    origin: Tuple[float, float]
    resolution: float
    size: Tuple[int, int]
    inflation: float


class GlobalPlanMessage(BaseModel):
    agents: List[PlayerName]
    priority: List[PlayerName]
    assignments: Mapping[PlayerName, List[str]]
    goal_to_cp: Mapping[str, str]
    goals: List[GoalSpec]
    collection_points: List[CollectionSpec]
    grid: GridSpec


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polygon_from_shape(shape) -> Polygon:
    if isinstance(shape, Polygon):
        return shape
    if isinstance(shape, LinearRing):
        return Polygon(shape)
    if hasattr(shape, "buffer"):
        return shape.buffer(0)
    raise ValueError(f"Unsupported shape type {type(shape)}")


def _extract_boundary(static_obstacles: Sequence[StaticObstacle]) -> Polygon:
    boundary_poly = None
    max_area = -1.0
    for sob in static_obstacles:
        poly = _polygon_from_shape(sob.shape)
        area = poly.area
        if area > max_area:
            max_area = area
            boundary_poly = poly
    if boundary_poly is None:
        raise RuntimeError("Failed to extract boundary polygon")
    return boundary_poly


class GridPlanner:
    def __init__(
        self,
        static_obstacles: Sequence[StaticObstacle],
        robot_radius: float,
        inflation: float,
        resolution: float,
    ):
        self.resolution = resolution
        self.inflation = robot_radius + inflation
        self.boundary = _extract_boundary(static_obstacles)
        self.boundary_inner = self.boundary.buffer(-self.inflation)
        obstacle_polys: List[Polygon] = []
        for sob in static_obstacles:
            poly = _polygon_from_shape(sob.shape)
            if poly.equals(self.boundary) or abs(poly.area - self.boundary.area) < 1e-6:
                continue
            obstacle_polys.append(poly)
        self.obstacle_polys = obstacle_polys
        self.obstacles_inflated = [poly.buffer(self.inflation) for poly in obstacle_polys]
        self.obs_index = STRtree(self.obstacles_inflated) if self.obstacles_inflated else None
        minx, miny, maxx, maxy = self.boundary.bounds
        self.origin = (float(minx), float(miny))
        self.size_x = int(math.ceil((maxx - minx) / self.resolution)) + 1
        self.size_y = int(math.ceil((maxy - miny) / self.resolution)) + 1
        self.occupancy = self._build_occupancy()

    def world_to_cell(self, point: Tuple[float, float]) -> Tuple[int, int]:
        ix = int((point[0] - self.origin[0]) / self.resolution)
        iy = int((point[1] - self.origin[1]) / self.resolution)
        return ix, iy

    def cell_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        x = self.origin[0] + (cell[0] + 0.5) * self.resolution
        y = self.origin[1] + (cell[1] + 0.5) * self.resolution
        return (x, y)

    def _in_bounds(self, cell: Tuple[int, int]) -> bool:
        ix, iy = cell
        return 0 <= ix < self.size_x and 0 <= iy < self.size_y

    def _is_free_cell(self, cell: Tuple[int, int]) -> bool:
        ix, iy = cell
        if not self._in_bounds(cell):
            return False
        return not self.occupancy[iy, ix]

    def _build_occupancy(self) -> np.ndarray:
        occ = np.zeros((self.size_y, self.size_x), dtype=bool)
        if self.boundary_inner.is_empty:
            occ[:] = True
            return occ
        for iy in range(self.size_y):
            y = self.origin[1] + (iy + 0.5) * self.resolution
            for ix in range(self.size_x):
                x = self.origin[0] + (ix + 0.5) * self.resolution
                p = Point(x, y)
                if not self.boundary_inner.contains(p):
                    occ[iy, ix] = True
                    continue
                if self.obs_index:
                    hits = self.obs_index.query(p)
                    if any(self.obstacles_inflated[int(h)].contains(p) or self.obstacles_inflated[int(h)].distance(p) < 1e-9 for h in hits):
                        occ[iy, ix] = True
        return occ

    def _nearest_free(self, cell: Tuple[int, int], search_radius: int = 8) -> Optional[Tuple[int, int]]:
        if self._is_free_cell(cell):
            return cell
        cx, cy = cell
        best: Optional[Tuple[int, int]] = None
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    cand = (cx + dx, cy + dy)
                    if self._is_free_cell(cand):
                        best = cand
                        break
                if best:
                    break
            if best:
                break
        return best

    def _line_is_free(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        seg = LineString([a, b])
        if self.boundary_inner.is_empty:
            return False
        if not self.boundary_inner.contains(seg):
            return False
        buffered = seg.buffer(self.inflation)
        if self.obs_index:
            hits = self.obs_index.query(buffered)
            if any(buffered.intersects(self.obstacles_inflated[int(h)]) for h in hits):
                return False
        return True

    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        diag_cost = math.sqrt(2)
        open_set: List[Tuple[float, Tuple[int, int]]] = []
        g: Dict[Tuple[int, int], float] = {start: 0.0}
        f: Dict[Tuple[int, int], float] = {start: _euclidean(start, goal)}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        heapq.heappush(open_set, (f[start], start))
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                break
            for dx, dy in nbrs:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._is_free_cell(nxt):
                    continue
                step = diag_cost if dx != 0 and dy != 0 else 1.0
                tentative = g[current] + step
                if tentative < g.get(nxt, 1e9):
                    came_from[nxt] = current
                    g[nxt] = tentative
                    f[nxt] = tentative + _euclidean(nxt, goal)
                    heapq.heappush(open_set, (f[nxt], nxt))
        if goal not in came_from and goal != start:
            return []
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def _prune(self, pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(pts) <= 2:
            return pts
        pruned = [pts[0]]
        idx = 0
        while idx < len(pts) - 1:
            j = len(pts) - 1
            while j > idx + 1:
                if self._line_is_free(pruned[-1], pts[j]):
                    break
                j -= 1
            pruned.append(pts[j])
            idx = j
        return pruned

    def plan(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        if self.boundary_inner.is_empty:
            return []
        start_cell = self._nearest_free(self.world_to_cell(start_xy))
        goal_cell = self._nearest_free(self.world_to_cell(goal_xy))
        if start_cell is None or goal_cell is None:
            return []
        cells = self._astar(start_cell, goal_cell)
        if not cells:
            return []
        pts = [self.cell_to_world(c) for c in cells]
        return self._prune(pts)

    def path_cost(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> float:
        path = self.plan(start_xy, goal_xy)
        if len(path) < 2:
            return _euclidean(start_xy, goal_xy) * 3.0
        return sum(_euclidean(a, b) for a, b in zip(path[:-1], path[1:]))


@dataclass(frozen=True)
class Pdm4arAgentParams:
    grid_resolution: float = 0.35
    inflation_margin: float = 0.25
    replan_period: float = 0.6
    waypoint_tolerance: float = 0.35
    lookahead: float = 0.9
    max_speed: float = 1.3
    max_omega: float = 1.8
    heading_gain: float = 2.0
    speed_gain: float = 1.1
    slow_radius: float = 1.2
    yield_distance: float = 1.5
    robot_safety: float = 0.9
    dynamic_stop: float = 0.8
    avoidance_horizon: float = 1.2
    stuck_timeout: float = 3.0
    stuck_epsilon: float = 0.15
    recovery_time: float = 1.2
    recovery_turn_rate: float = 0.8
    recovery_reverse_speed: float = 0.2
    route_depth: int = 5
    route_beam_width: int = 5
    route_time_budget: float = 85.0
    single_agent_inflation: float = 0.2
    reverse_threshold: float = 0.95
    reverse_speed_factor: float = 0.65


PASS_SIDE_HOLD_SEC = 2.0
OMEGA_BIAS = 0.35
SIDESTEP_DIST = 0.8
SIDESTEP_TIMEOUT = 1.0
YIELD_MAX_SEC = 2.0
PROGRESS_EPS = 0.25
NO_PROGRESS_TIMEOUT = 12.0


class AgentMode(Enum):
    SEEK_GOAL = auto()
    DELIVER = auto()
    YIELD = auto()
    RECOVERY = auto()
    IDLE = auto()


def _priority_index(order: List[PlayerName]) -> Dict[PlayerName, int]:
    return {name: idx for idx, name in enumerate(order)}


class Pdm4arAgent(Agent):
    name: PlayerName
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: DiffDriveGeometry
    sp: DiffDriveParameters

    def __init__(self):
        self.params = Pdm4arAgentParams()
        self.grid_planner: Optional[GridPlanner] = None
        self.rng = np.random.default_rng()
        self.goals: Dict[str, GoalSpec] = {}
        self.collection_points: Dict[str, CollectionSpec] = {}
        self.goal_to_cp: Dict[str, str] = {}
        self.assignments: Dict[PlayerName, List[str]] = {}
        self.priority_order: List[PlayerName] = []
        self.priority_map: Dict[PlayerName, int] = {}
        self.current_goal: Optional[str] = None
        self.completed_goals: set[str] = set()
        self.lost_goals: set[str] = set()
        self.mode: AgentMode = AgentMode.IDLE
        self.current_path: List[Tuple[float, float]] = []
        self.waypoint_idx: int = 0
        self.last_plan_time: float = -1.0
        self.last_progress_time: float = 0.0
        self.last_progress_dist: float = 1e9
        self.pass_side: int = 1
        self.pass_side_until: float = 0.0
        self.blocked_by_dyn_since: Optional[float] = None
        self.yield_since: Optional[float] = None
        self.recovery_cooldown_until: float = 0.0
        self.omega_prev: float = 0.0
        self.last_target_dist: float = 1e9
        self.last_target_progress_time: float = 0.0
        self.target_cooldown: Dict[str, float] = {}
        self.recovery_until: float = 0.0
        self.recovery_dir: float = 1.0
        self.last_cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)
        self.last_events: deque[str] = deque(maxlen=5)
        self.last_carried: Optional[str] = None
        self.single_agent: bool = False

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        self.name = init_sim_obs.my_name
        self.goal = init_sim_obs.goal
        self.static_obstacles = init_sim_obs.dg_scenario.static_obstacles
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        seed = init_sim_obs.seed + hash(self.name) % 997
        self.rng = np.random.default_rng(seed)
        self._log_event("episode_init")

    def on_receive_global_plan(self, serialized_msg: str):
        global_plan = GlobalPlanMessage.model_validate_json(serialized_msg)
        self.assignments = {k: list(v) for k, v in global_plan.assignments.items()}
        self.priority_order = list(global_plan.priority)
        self.priority_map = _priority_index(self.priority_order)
        self.goal_to_cp = dict(global_plan.goal_to_cp)
        self.goals = {g.goal_id: g for g in global_plan.goals}
        self.collection_points = {c.point_id: c for c in global_plan.collection_points}
        self.single_agent = len(global_plan.agents) == 1
        if len(global_plan.agents) == 1:
            self.params = replace(
                self.params,
                max_speed=1.8,
                speed_gain=1.5,
                slow_radius=0.7,
                heading_gain=2.0,
                max_omega=2.5,
                lookahead=0.6,
                waypoint_tolerance=0.22,
                replan_period=0.4,
                robot_safety=1.1,
            )
        else:
            self.params = replace(
                self.params,
                max_speed=1.0,
                speed_gain=0.95,
                heading_gain=1.6,
                slow_radius=1.5,
                lookahead=0.7,
                replan_period=0.5,
                yield_distance=2.8,
                dynamic_stop=1.4,
                robot_safety=1.35,
                avoidance_horizon=1.7,
            )
        inflation = global_plan.grid.inflation - self.sg.radius
        if self.single_agent:
            inflation = max(self.params.single_agent_inflation, 0.05)
        else:
            inflation = max(inflation, 0.05)
        self.grid_planner = GridPlanner(
            static_obstacles=self.static_obstacles,
            robot_radius=self.sg.radius,
            inflation=inflation,
            resolution=global_plan.grid.resolution,
        )
        self.mode = AgentMode.SEEK_GOAL
        self._log_event("plan_received")

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        t = float(sim_obs.time)
        my_state: DiffDriveState = sim_obs.players[self.name].state
        carried_goal = sim_obs.players[self.name].collected_goal_id
        base_mode = AgentMode.DELIVER if carried_goal is not None else AgentMode.SEEK_GOAL

        if self.last_carried is not None and carried_goal is None:
            if self.current_goal is not None:
                self.completed_goals.add(self.current_goal)
            self.current_goal = None
            self.mode = AgentMode.SEEK_GOAL
            base_mode = AgentMode.SEEK_GOAL
            target_changed = True
        else:
            target_changed = False
            if carried_goal is None and self.current_goal in self.completed_goals:
                self.current_goal = None

        prev_mode = self.mode
        if carried_goal is not None and carried_goal != self.current_goal:
            self.current_goal = carried_goal
            target_changed = True

        if self.mode == AgentMode.RECOVERY and t < self.recovery_until:
            pass
        elif self.mode == AgentMode.YIELD:
            pass
        else:
            if self.mode == AgentMode.RECOVERY and t >= self.recovery_until:
                target_changed = True
            self.mode = base_mode
        if self.mode == AgentMode.DELIVER and prev_mode != AgentMode.DELIVER:
            target_changed = True
        self.last_carried = carried_goal

        if base_mode == AgentMode.SEEK_GOAL and self.current_goal and self._goal_missing(sim_obs, my_state):
            self.lost_goals.add(self.current_goal)
            self._log_event(f"goal_lost:{self.current_goal}")
            self.current_goal = None
            target_changed = True

        if base_mode == AgentMode.SEEK_GOAL and (
            self.current_goal is None or self.current_goal in self.completed_goals
        ):
            chosen = self._pick_next_goal(my_state, t, sim_obs)
            if chosen:
                self.current_goal = chosen
                target_changed = True
                self._log_event(f"target:{chosen}")

        if carried_goal is None and base_mode == AgentMode.SEEK_GOAL and prev_mode == AgentMode.DELIVER:
            if self.current_goal is not None:
                self.completed_goals.add(self.current_goal)
            self.current_goal = None
            self.mode = AgentMode.SEEK_GOAL
            target_changed = True
            self._log_event("delivered")

        need_plan = target_changed or self.current_path == []
        if t - self.last_plan_time > self.params.replan_period:
            need_plan = True
        target_point = self._target_point()
        if need_plan and self.mode in (AgentMode.SEEK_GOAL, AgentMode.DELIVER, AgentMode.YIELD):
            if target_point is not None and self.grid_planner is not None:
                self.current_path = self.grid_planner.plan(
                    (my_state.x, my_state.y),
                    target_point,
                )
                self.waypoint_idx = 0
                self.last_plan_time = t
                self.last_progress_time = t
                self.last_progress_dist = 1e9

        if target_point is not None and self.current_goal is not None:
            target_dist = _euclidean((my_state.x, my_state.y), target_point)
            if target_changed or self.last_target_progress_time == 0.0:
                self.last_target_progress_time = t
                self.last_target_dist = target_dist
            elif target_dist < self.last_target_dist - PROGRESS_EPS:
                self.last_target_progress_time = t
                self.last_target_dist = target_dist
            elif t - self.last_target_progress_time > NO_PROGRESS_TIMEOUT:
                if base_mode == AgentMode.DELIVER:
                    if t >= self.recovery_cooldown_until:
                        self._enter_recovery(t)
                    self.last_target_progress_time = t
                    self.last_target_dist = target_dist
                    target_changed = True
                    self._log_event("deliver_recover")
                else:
                    self.target_cooldown[self.current_goal] = t + 10.0
                    moved = False
                    if self.name in self.assignments and self.current_goal in self.assignments[self.name]:
                        try:
                            self.assignments[self.name].remove(self.current_goal)
                            moved = True
                        except ValueError:
                            moved = False
                        if moved:
                            self.assignments[self.name].append(self.current_goal)
                    if not moved:
                        self.lost_goals.add(self.current_goal)
                    self.current_goal = None
                    self.current_path = []
                    self.waypoint_idx = 0
                    base_mode = AgentMode.SEEK_GOAL
                    self.mode = AgentMode.SEEK_GOAL
                    self.last_target_dist = 1e9
                    self.last_target_progress_time = t
                    target_changed = True
                    self._log_event("switch_goal")
        else:
            self.last_target_dist = 1e9

        min_dyn, _, _, _, _ = self._dynamic_clearance((my_state.x, my_state.y), my_state.psi, sim_obs)
        if self.current_path and self.waypoint_idx < len(self.current_path):
            wp = self.current_path[self.waypoint_idx]
            dist_wp = _euclidean((my_state.x, my_state.y), wp)
            if dist_wp < self.params.waypoint_tolerance and self.waypoint_idx < len(self.current_path) - 1:
                self.waypoint_idx += 1
                self._log_event("advance_wp")
            if dist_wp < self.last_progress_dist - self.params.stuck_epsilon:
                self.last_progress_time = t
                self.last_progress_dist = dist_wp
            elif t - self.last_progress_time > self.params.stuck_timeout and self.mode not in (
                AgentMode.RECOVERY,
                AgentMode.YIELD,
            ):
                if min_dyn < self.params.yield_distance:
                    if self.mode != AgentMode.YIELD:
                        self.mode = AgentMode.YIELD
                        if self.blocked_by_dyn_since is None:
                            self.blocked_by_dyn_since = t
                        if self.yield_since is None:
                            self.yield_since = t
                    self._log_event("stuck_dyn")
                elif t >= self.recovery_cooldown_until:
                    self._enter_recovery(t)
                    self._log_event("stuck_static")
                else:
                    self.last_progress_time = t
                    self.last_progress_dist = dist_wp
        else:
            self.last_progress_time = t
            self.last_progress_dist = 1e9

        if self.mode == AgentMode.RECOVERY:
            cmd = self._recovery_command(my_state, t)
        elif self.current_path:
            cmd = self._track_path(my_state, sim_obs, t)
        else:
            cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)

        self.last_cmd = cmd
        return cmd

    def _goal_missing(self, sim_obs: SimObservations, my_state: DiffDriveState) -> bool:
        if self.current_goal is None:
            return False
        goal_spec = self.goals.get(self.current_goal)
        if goal_spec is None:
            return True
        dist = _euclidean((my_state.x, my_state.y), goal_spec.center)
        if dist > 2.0:
            return False
        if sim_obs.available_goals is None:
            return False
        if self.current_goal in sim_obs.available_goals:
            return False
        return True

    def _pick_next_goal(self, my_state: DiffDriveState, t: float, sim_obs: Optional[SimObservations] = None) -> Optional[str]:
        if not self.grid_planner:
            return None
        expired = [gid for gid, until in list(self.target_cooldown.items()) if until <= t]
        for gid in expired:
            self.target_cooldown.pop(gid, None)
            self.lost_goals.discard(gid)

        carried_by_others: set[str] = set()
        if sim_obs is not None:
            for other, obs in sim_obs.players.items():
                if other == self.name:
                    continue
                if obs.collected_goal_id:
                    carried_by_others.add(obs.collected_goal_id)

        def _ready(gid: str) -> bool:
            return self.target_cooldown.get(gid, 0.0) <= t
        ordered_assigned = [
            gid
            for gid in self.assignments.get(self.name, [])
            if gid not in self.completed_goals
            and gid not in self.lost_goals
            and gid not in carried_by_others
            and _ready(gid)
        ]
        if ordered_assigned:
            return ordered_assigned[0]
        candidates = [
            gid
            for gid in self.goals
            if gid not in self.completed_goals
            and gid not in self.lost_goals
            and gid not in carried_by_others
            and _ready(gid)
        ]
        if not candidates:
            return None
        best = None
        best_cost = 1e9
        for gid in sorted(candidates):
            goal = self.goals[gid]
            cost = self.grid_planner.path_cost((my_state.x, my_state.y), goal.center)
            if cost < best_cost:
                best_cost = cost
                best = gid
        return best

    def _target_point(self) -> Optional[Tuple[float, float]]:
        if self.current_goal is None:
            return None
        mode = self.mode
        if mode == AgentMode.YIELD and self.last_carried is not None:
            mode = AgentMode.DELIVER
        if mode == AgentMode.DELIVER:
            cp_id = self.goal_to_cp.get(self.current_goal)
            cp = self.collection_points.get(cp_id) if cp_id else None
            if cp is None and self.collection_points:
                goal = self.goals.get(self.current_goal)
                anchor = goal.center if goal else (0.0, 0.0)
                cp = min(self.collection_points.values(), key=lambda c: _euclidean(anchor, c.center))
            return cp.center if cp else None
        goal = self.goals.get(self.current_goal)
        return goal.center if goal else None

    def _track_path(self, my_state: DiffDriveState, sim_obs: SimObservations, t: float) -> DiffDriveCommands:
        if not self.current_path:
            return self._to_wheel_commands(0.0, 0.0)
        path_len = len(self.current_path)
        wp_idx = min(self.waypoint_idx, path_len - 1)
        wp = self.current_path[wp_idx]
        pos = (my_state.x, my_state.y)
        to_wp = (wp[0] - pos[0], wp[1] - pos[1])
        dist = math.hypot(*to_wp)

        tolerance = self.params.waypoint_tolerance
        if wp_idx == path_len - 1:
            radius = None
            if self.mode == AgentMode.DELIVER:
                cp_id = self.goal_to_cp.get(self.current_goal)
                cp = self.collection_points.get(cp_id) if cp_id else None
                radius = cp.radius if cp else None
            else:
                goal_spec = self.goals.get(self.current_goal) if self.current_goal else None
                radius = goal_spec.radius if goal_spec else None
            if radius is not None:
                tolerance = min(tolerance, max(0.1, radius * 0.8))

        if dist < tolerance:
            if wp_idx < path_len - 1:
                self.waypoint_idx = wp_idx + 1
                wp = self.current_path[self.waypoint_idx]
                to_wp = (wp[0] - pos[0], wp[1] - pos[1])
                dist = math.hypot(*to_wp)
            else:
                self.current_path = []
                self.waypoint_idx = 0
                return self._to_wheel_commands(0.0, 0.0)

        desired_heading = math.atan2(to_wp[1], to_wp[0])
        heading_err = _wrap_angle(desired_heading - my_state.psi)

        look_wp = wp
        accum = 0.0
        for idx in range(self.waypoint_idx, len(self.current_path) - 1):
            seg = _euclidean(self.current_path[idx], self.current_path[idx + 1])
            accum += seg
            look_wp = self.current_path[idx + 1]
            if accum >= self.params.lookahead:
                break
        if look_wp != wp:
            vec = (look_wp[0] - pos[0], look_wp[1] - pos[1])
            desired_heading = math.atan2(vec[1], vec[0])
            heading_err = _wrap_angle(desired_heading - my_state.psi)
            dist = math.hypot(*vec)

        abs_heading_err = abs(heading_err)

        if abs_heading_err > math.pi * 0.5:
            v_des = 0.0
            omega_des = math.copysign(self.params.max_omega, heading_err)

        elif abs_heading_err > math.pi / 3:
            v_des = self.params.max_speed * 0.15
            omega_des = math.copysign(self.params.max_omega * 0.9, heading_err)

        elif abs_heading_err > math.pi / 6:
            v_des = min(self.params.max_speed * 0.5, self.params.speed_gain * dist * 0.5)
            if dist < self.params.slow_radius:
                v_des *= dist / max(self.params.slow_radius, 1e-3)
            omega_des = self.params.heading_gain * heading_err * 1.2
            omega_des = max(-self.params.max_omega, min(self.params.max_omega, omega_des))

        else:
            v_des = min(self.params.max_speed, self.params.speed_gain * dist)
            cos_scale = max(0.0, min(1.0, math.cos(heading_err)))
            v_des *= max(0.4, cos_scale)
            if dist < self.params.slow_radius:
                v_des *= dist / max(self.params.slow_radius, 1e-3)
            omega_des = self.params.heading_gain * heading_err
            omega_des = max(-self.params.max_omega, min(self.params.max_omega, omega_des))

        if self.mode == AgentMode.DELIVER and not self.single_agent:
            v_des = min(v_des * 1.1, self.params.max_speed * 1.1)

        omega_des = 0.7 * self.omega_prev + 0.3 * omega_des
        self.omega_prev = omega_des

        v_des, omega_des = self._apply_avoidance(
            v_des, omega_des, pos, my_state.psi, sim_obs, t
        )
        return self._to_wheel_commands(v_des, omega_des)

    def _apply_avoidance(
        self,
        v: float,
        omega: float,
        pos: Tuple[float, float],
        heading: float,
        sim_obs: SimObservations,
        t: float,
    ) -> Tuple[float, float]:
        base_mode = AgentMode.DELIVER if sim_obs.players[self.name].collected_goal_id else AgentMode.SEEK_GOAL
        min_static = self._min_static_distance(pos)
        stop_margin = self.params.robot_safety * (0.5 if self.single_agent else 0.4)
        slow_margin = self.params.robot_safety
        if min_static < stop_margin:
            v = 0.0
            omega = math.copysign(self.params.max_omega * 0.5, self.rng.choice([-1, 1]))
            self._log_event("static_stop")
        elif min_static < slow_margin:
            slow_factor = 0.7 if self.single_agent else 0.55
            v *= slow_factor

        min_dyn, close_dir, higher_priority, closest_name, rel_angle = self._dynamic_clearance(
            pos, heading, sim_obs
        )

        if min_dyn < self.params.yield_distance and rel_angle is not None:
            if t > self.pass_side_until:
                if abs(rel_angle) > 0.05:
                    self.pass_side = -1 if rel_angle > 0 else 1
                else:
                    self.pass_side = -1 if (hash(self.name) % 2 == 0) else 1
                self.pass_side_until = t + PASS_SIDE_HOLD_SEC

        in_yield = self.mode == AgentMode.YIELD
        if not in_yield:
            self.yield_since = None

        hard_dyn_stop = min(self.params.dynamic_stop, self.params.robot_safety)
        if min_dyn < hard_dyn_stop:
            v = 0.0
            omega = self.pass_side * self.params.max_omega * 0.3
            if higher_priority and not in_yield:
                self.mode = AgentMode.YIELD
                in_yield = True
                self.blocked_by_dyn_since = self.blocked_by_dyn_since or t
                self.yield_since = self.yield_since or t
            self._log_event("dyn_stop")
        elif min_dyn < self.params.dynamic_stop:
            if higher_priority:
                if not in_yield:
                    self.mode = AgentMode.YIELD
                    in_yield = True
                    self.blocked_by_dyn_since = self.blocked_by_dyn_since or t
                    self.yield_since = self.yield_since or t
                v = 0.0
                omega = self.pass_side * self.params.max_omega * 0.35
                self._log_event("yield_stop")
            else:
                v = min(v * 0.2, self.params.max_speed * 0.2)
                omega += self.pass_side * OMEGA_BIAS
                self._log_event("prio_go")
        elif higher_priority and min_dyn < self.params.yield_distance:
            v *= 0.25
            omega += self.pass_side * OMEGA_BIAS
            if not in_yield and (t - self.last_progress_time > self.params.stuck_timeout * 0.5):
                self.mode = AgentMode.YIELD
                in_yield = True
                self.blocked_by_dyn_since = self.blocked_by_dyn_since or t
                self.yield_since = self.yield_since or t
            self._log_event("yield")
        elif min_dyn < self.params.yield_distance:
            omega += self.pass_side * OMEGA_BIAS * 0.5

        sidestepping = False
        if self.mode == AgentMode.YIELD:
            if self.yield_since is None:
                self.yield_since = t
            if self.blocked_by_dyn_since is None:
                self.blocked_by_dyn_since = t
            clear_enough = min_dyn > (self.params.yield_distance + 0.3) and (
                self.yield_since is None or t - self.yield_since > 0.3
            )
            timeout = self.yield_since is not None and t - self.yield_since > YIELD_MAX_SEC
            if clear_enough or timeout:
                self.mode = base_mode
                self.blocked_by_dyn_since = None
                self.yield_since = None
                if timeout:
                    v *= 0.6
                    omega += self.pass_side * OMEGA_BIAS
            elif (
                min_dyn < self.params.yield_distance
                and self.blocked_by_dyn_since is not None
                and t - self.blocked_by_dyn_since > 1.5
            ):
                step_len = SIDESTEP_DIST
                left_normal = (-math.sin(heading), math.cos(heading))
                sidestep_target: Optional[Tuple[float, float]] = None
                for _ in range(3):
                    cand = (
                        pos[0] + self.pass_side * step_len * left_normal[0],
                        pos[1] + self.pass_side * step_len * left_normal[1],
                    )
                    if self.grid_planner and self.grid_planner.boundary.contains(Point(cand)):
                        if self._min_static_distance(cand) > self.params.robot_safety * 0.6:
                            sidestep_target = cand
                            break
                    step_len *= 0.5
                if sidestep_target is not None:
                    sidestep_window = self.blocked_by_dyn_since + 1.5 + SIDESTEP_TIMEOUT
                    vec = (sidestep_target[0] - pos[0], sidestep_target[1] - pos[1])
                    dist_step = math.hypot(*vec)
                    if t < sidestep_window:
                        desired = math.atan2(vec[1], vec[0])
                        err = _wrap_angle(desired - heading)
                        omega = max(
                            -self.params.max_omega * 0.5,
                            min(self.params.max_omega * 0.5, self.params.heading_gain * err * 0.5),
                        )
                        v = min(self.params.max_speed * 0.35, dist_step)
                        sidestepping = True
                        if dist_step < 0.15:
                            self.mode = base_mode
                            self.blocked_by_dyn_since = None
                            self.yield_since = None
                    else:
                        self.mode = base_mode
                        self.blocked_by_dyn_since = None
                        self.yield_since = None
                        sidestepping = False

        if self.mode == AgentMode.YIELD and not sidestepping and min_dyn < self.params.yield_distance:
            v = 0.0 if higher_priority else v * 0.4
            omega += self.pass_side * OMEGA_BIAS * 0.8

        if self.mode != AgentMode.YIELD and min_dyn > self.params.yield_distance + 0.5:
            self.blocked_by_dyn_since = None

        horizon = max(0.2, min(1.0, v * 1.5))
        ahead = (pos[0] + horizon * math.cos(heading), pos[1] + horizon * math.sin(heading))
        if self.grid_planner and not self.grid_planner.boundary.contains(Point(ahead)):
            v = 0.0
            omega = math.copysign(self.params.max_omega * 0.4, omega or 1.0)
            self._log_event("edge_stop")

        return v, omega

    def _min_static_distance(self, pos: Tuple[float, float]) -> float:
        p = Point(pos)
        distances = [self.grid_planner.boundary.exterior.distance(p)] if self.grid_planner else []
        if self.grid_planner:
            for poly in self.grid_planner.obstacle_polys:
                distances.append(p.distance(poly))
        return min(distances) if distances else 10.0

    def _dynamic_clearance(
        self, pos: Tuple[float, float], heading: float, sim_obs: SimObservations
    ) -> Tuple[float, float, bool, Optional[PlayerName], Optional[float]]:
        min_dist = 10.0
        dir_sign = 0.0
        higher_prio = False
        closest_name: Optional[PlayerName] = None
        rel_angle: Optional[float] = None
        self_carrying = sim_obs.players[self.name].collected_goal_id is not None
        for other, obs in sim_obs.players.items():
            if other == self.name:
                continue
            op = obs.state
            d = _euclidean(pos, (op.x, op.y))
            if d < min_dist:
                min_dist = d
                rel_angle = _wrap_angle(math.atan2(op.y - pos[1], op.x - pos[0]) - heading)
                dir_sign = -1.0 if rel_angle > 0 else 1.0
                other_carrying = obs.collected_goal_id is not None
                if other_carrying and not self_carrying:
                    higher_prio = True
                elif self_carrying and not other_carrying:
                    higher_prio = False
                else:
                    higher_prio = self.priority_map.get(other, 0) < self.priority_map.get(self.name, 0)
                closest_name = other
        return min_dist, dir_sign, higher_prio, closest_name, rel_angle

    def _to_wheel_commands(self, v: float, omega: float) -> DiffDriveCommands:
        L = self.sg.wheelbase
        r = self.sg.wheelradius
        omega_r = (2 * v + omega * L) / (2 * r)
        omega_l = (2 * v - omega * L) / (2 * r)
        limit_low, limit_high = self.sp.omega_limits
        omega_r = max(limit_low, min(limit_high, omega_r))
        omega_l = max(limit_low, min(limit_high, omega_l))
        return DiffDriveCommands(omega_l=omega_l, omega_r=omega_r)

    def _enter_recovery(self, t: float):
        self.mode = AgentMode.RECOVERY
        self.recovery_until = t + self.params.recovery_time
        self.recovery_dir = 1.0 if (hash(self.name) % 2 == 0) else -1.0
        self.current_path = []
        self.waypoint_idx = 0
        self.blocked_by_dyn_since = None
        self.yield_since = None

    def _recovery_command(self, my_state: DiffDriveState, t: float) -> DiffDriveCommands:
        remaining = max(0.0, self.recovery_until - t)
        omega = self.recovery_dir * self.params.recovery_turn_rate
        v = -self.params.recovery_reverse_speed if remaining > self.params.recovery_time * 0.5 else 0.0
        if remaining <= 0:
            self.mode = AgentMode.SEEK_GOAL
            self.last_progress_time = t
            self.last_progress_dist = 1e9
            self.recovery_cooldown_until = t + 2.0
        return self._to_wheel_commands(v, omega)

    def _log_event(self, msg: str):
        self.last_events.append(msg)

    def on_get_extra(self):
        return {
            "mode": self.mode.name,
            "target": self.current_goal,
            "path": self.current_path,
            "wp_idx": self.waypoint_idx,
            "events": list(self.last_events),
            "last_cmd": (self.last_cmd.omega_l, self.last_cmd.omega_r),
        }


class Pdm4arGlobalPlanner(GlobalPlanner):
    def __init__(self):
        self.params = Pdm4arAgentParams()

    def _goal_specs(self, shared_goals: Mapping[str, object]) -> List[GoalSpec]:
        specs: List[GoalSpec] = []
        for gid, goal in shared_goals.items():
            poly = goal.polygon
            center = (float(poly.centroid.x), float(poly.centroid.y))
            radius = float(poly.minimum_clearance / 2) if hasattr(poly, "minimum_clearance") else 0.3
            specs.append(GoalSpec(goal_id=gid, center=center, radius=radius))
        return specs

    def _cp_specs(self, cps: Mapping[str, object]) -> List[CollectionSpec]:
        specs: List[CollectionSpec] = []
        for cid, cp in cps.items():
            poly = cp.polygon
            center = (float(poly.centroid.x), float(poly.centroid.y))
            radius = float(poly.minimum_clearance / 2) if hasattr(poly, "minimum_clearance") else 0.8
            specs.append(CollectionSpec(point_id=cid, center=center, radius=radius))
        return specs

    def _plan_agent_route(
        self,
        start_id: object,
        goal_ids: Sequence[str],
        cp_ids: Sequence[str],
        path_cost,
        budget: float,
    ) -> List[Tuple[str, str]]:
        if not cp_ids:
            return []
        max_depth = min(self.params.route_depth, len(goal_ids))
        beam = max(1, self.params.route_beam_width)
        best_route: List[Tuple[str, str]] = []
        best_cost = float("inf")
        best_delivered = 0

        def dfs(pos_id: object, remaining: set[str], route: List[Tuple[str, str]], cost: float, depth: int):
            nonlocal best_route, best_cost, best_delivered
            delivered = len(route)
            if delivered > best_delivered or (delivered == best_delivered and cost < best_cost):
                best_route = list(route)
                best_cost = cost
                best_delivered = delivered
            if depth >= max_depth or not remaining:
                return
            ordered_goals = sorted(remaining, key=lambda g: path_cost(pos_id, g))
            for gid in ordered_goals[:beam]:
                travel_to_goal = path_cost(pos_id, gid)
                if math.isinf(travel_to_goal):
                    continue
                for cp_id in cp_ids:
                    to_cp = path_cost(gid, cp_id)
                    if math.isinf(to_cp):
                        continue
                    new_cost = cost + travel_to_goal + to_cp
                    if delivered >= 1 and new_cost > budget:
                        continue
                    route.append((gid, cp_id))
                    dfs(cp_id, remaining - {gid}, route, new_cost, depth + 1)
                    route.pop()

        dfs(start_id, set(goal_ids), [], 0.0, 0)
        return best_route

    def _order_route(
        self,
        start: object,
        goal_list: Sequence[str],
        goal_to_cp: Mapping[str, str],
        path_cost,
        coord_lookup: Mapping[object, Tuple[float, float]],
        farthest_first: bool = False,
    ) -> List[str]:
        ordered: List[str] = []
        pos: object = start
        remaining = set(goal_list)
        while remaining:
            best_gid = None
            best_cost = float("-inf") if farthest_first else float("inf")
            for gid in remaining:
                cp_id = goal_to_cp.get(gid)
                cp_cost = path_cost(gid, cp_id) if cp_id in coord_lookup else 0.0
                c = path_cost(pos, gid) + cp_cost
                if (farthest_first and c > best_cost) or (not farthest_first and c < best_cost):
                    best_cost = c
                    best_gid = gid
            if best_gid is None:
                break
            ordered.append(best_gid)
            remaining.remove(best_gid)
            next_pos = goal_to_cp.get(best_gid)
            pos = next_pos if next_pos in coord_lookup else best_gid
        return ordered

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        t0 = time.perf_counter()
        players = sorted(init_sim_obs.players_obs.keys())
        priority = list(players)
        static_obstacles = init_sim_obs.dg_scenario.static_obstacles
        robot_radius = next(iter(init_sim_obs.players_obs.values())).model_geometry.radius
        grid_planner = GridPlanner(
            static_obstacles=static_obstacles,
            robot_radius=robot_radius,
            inflation=self.params.inflation_margin,
            resolution=self.params.grid_resolution,
        )

        goal_specs = self._goal_specs(init_sim_obs.shared_goals) if init_sim_obs.shared_goals else []
        cp_specs = self._cp_specs(init_sim_obs.collection_points) if init_sim_obs.collection_points else []
        goal_lookup = {g.goal_id: g for g in goal_specs}
        cp_lookup = {c.point_id: c for c in cp_specs}

        coord_lookup: Dict[object, Tuple[float, float]] = {}
        coord_lookup.update({g.goal_id: g.center for g in goal_specs})
        coord_lookup.update({c.point_id: c.center for c in cp_specs})
        coord_lookup.update(
            {p: (init_sim_obs.initial_states[p].x, init_sim_obs.initial_states[p].y) for p in players}
        )
        cost_cache: Dict[Tuple[object, object], float] = {}

        def _coord(node: object) -> Tuple[float, float]:
            if isinstance(node, tuple):
                return node
            if node in coord_lookup:
                return coord_lookup[node]
            raise KeyError(node)

        def path_cost(a: object, b: object) -> float:
            try:
                key = (a, b)
                if key not in cost_cache:
                    cost_cache[key] = grid_planner.path_cost(_coord(a), _coord(b))
                return cost_cache[key]
            except KeyError:
                return float("inf")

        goal_to_cp: Dict[str, str] = {}
        for gid, goal in goal_lookup.items():
            if cp_specs:
                cp_id = min(cp_specs, key=lambda c: path_cost(gid, c.point_id)).point_id
                goal_to_cp[gid] = cp_id

        assignments: Dict[PlayerName, List[str]] = {p: [] for p in players}
        current_pos: Dict[PlayerName, Tuple[float, float]] = {
            p: (init_sim_obs.initial_states[p].x, init_sim_obs.initial_states[p].y) for p in players
        }
        remaining = set(goal_lookup.keys())
        while remaining:
            best: Optional[Tuple[PlayerName, str, float]] = None
            for gid in sorted(remaining):
                for p in players:
                    cp_id = goal_to_cp.get(gid)
                    cp_cost = path_cost(gid, cp_id) if cp_id in coord_lookup else 0.0
                    c = path_cost(current_pos[p], gid) + cp_cost
                    c += len(assignments[p]) * 1.0
                    key = (c, p, gid)
                    if best is None or key < (best[2], best[0], best[1]):
                        best = (p, gid, c)
            if best is None:
                break
            p, gid, _ = best
            assignments[p].append(gid)
            cp_id = goal_to_cp.get(gid)
            next_pos = goal_lookup[gid].center
            if cp_id and cp_id in cp_lookup:
                next_pos = cp_lookup[cp_id].center
            current_pos[p] = next_pos
            remaining.remove(gid)

        cp_ids = [c.point_id for c in cp_specs]
        budget = self.params.max_speed * self.params.route_time_budget
        for p, goal_list in assignments.items():
            if not goal_list:
                continue
            planned = self._plan_agent_route(
                start_id=p,
                goal_ids=goal_list,
                cp_ids=cp_ids,
                path_cost=path_cost,
                budget=budget,
            )
            if planned:
                for gid, cp_id in planned:
                    goal_to_cp[gid] = cp_id
                delivered_goals = [g for g, _ in planned]
                remaining_goals = [g for g in goal_list if g not in delivered_goals]
                tail_start = planned[-1][1] if planned else p
                tail = self._order_route(
                    start=tail_start,
                    goal_list=remaining_goals,
                    goal_to_cp=goal_to_cp,
                    path_cost=path_cost,
                    coord_lookup=coord_lookup,
                    farthest_first=True,
                )
                assignments[p] = delivered_goals + tail
            else:
                assignments[p] = self._order_route(
                    start=p,
                    goal_list=goal_list,
                    goal_to_cp=goal_to_cp,
                    path_cost=path_cost,
                    coord_lookup=coord_lookup,
                    farthest_first=True,
                )

        grid_spec = GridSpec(
            origin=grid_planner.origin,
            resolution=self.params.grid_resolution,
            size=(grid_planner.size_x, grid_planner.size_y),
            inflation=grid_planner.inflation,
        )
        plan = GlobalPlanMessage(
            agents=players,
            priority=priority,
            assignments=assignments,
            goal_to_cp=goal_to_cp,
            goals=goal_specs,
            collection_points=cp_specs,
            grid=grid_spec,
        )
        _ = time.perf_counter() - t0
        return plan.model_dump_json(round_trip=True)
