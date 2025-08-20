import os, json, math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import pandas as pd

# --------------------------- CONFIGURATION CLASSES ---------------------------

@dataclass
class Sim3DConfig:
    az_fov_deg: float = 180.0    # Horizontal field of view (degrees) or Horizontal sweep range
    n_az: int = 181              # Number of azimuth beams (This can be changes later for now I've kept it as 181 * 16)
    el_fov_deg: float = 30.0     # Vertical field of view (degrees) or Vertical sweep range
    n_el: int = 16               # Number of elevation beams
    max_range: float = 30.0      # Max LiDAR detection range (meters)
    dt: float = 0.1              # Time step between scans (seconds)
    vel_eps: float = 0.05        # Min velocity change for classifying approach/recede
    risk_dist_thresh: float = 3.0
    ttc_thresh: float = 2.0      # Time-to-collision threshold (seconds)

@dataclass
class Obstacle3D:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float # x,y,z are the distances or radii to the point. The vx, vy, vz are the velocities.
    radius: float
    def step(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

@dataclass
class Scenario3D:
    obstacles: List[Obstacle3D]
    t_steps: int

# ---------------------------
# --------------------------- LIDAR SCAN FUNCTIONS ---------------------------

def angle_grids(cfg: Sim3DConfig):

# This calulates the grid of angles both horizontal and vertical that the scan would use.

    az_start = -cfg.az_fov_deg / 2.0 # This ensures that the scan starts from the leftmost edge of the FOV
    el_start = -cfg.el_fov_deg / 2.0 # This ensures that the scan starts from the bottommost edge of the FOV
    az_step = cfg.az_fov_deg / (cfg.n_az - 1) if cfg.n_az > 1 else 0.0
    el_step = cfg.el_fov_deg / (cfg.n_el - 1) if cfg.n_el > 1 else 0.0
    az_list = [az_start + i * az_step for i in range(cfg.n_az)]
    el_list = [el_start + j * el_step for j in range(cfg.n_el)]
    return az_list, el_list

def direction_from_angles(az_deg: float, el_deg: float):

# This function takes the azimuth and elevation angles in degrees and converts them into a 3D unit direction vector 
# Formula used: TO convert degrees to radians:
# rad = deg * pi/180
# TO find componenets of the 3D unit vector:
# x = cos(elevation) * cos(azimuth)
# y = cos(elevation) * sin(azimuth)
# z = sin(elevation)

    az = math.radians(az_deg)
    el = math.radians(el_deg)
    cx = math.cos(el) * math.cos(az) # X-component
    cy = math.cos(el) * math.sin(az) # Y-component
    cz = math.sin(el) # Z-component
    return cx, cy, cz

def ray_sphere_intersection(o, d, c, r):

# This function checks if the ray is intersecting with a sphere, if yes it returns the intersection point
# o is the origin of the ray, d is the direction of the ray(unit vector), c center of the sphere, r radius of the sphere
# 
# The B and C values are from the ray and sphere equations
# If the disc values is < 0 there is no intersection and the ray misses the sphere 
  
    ox, oy, oz = o; dx, dy, dz = d; cx, cy, cz = c
    mx, my, mz = ox - cx, oy - cy, oz - cz
    B = 2.0 * (dx * mx + dy * my + dz * mz)
    C = mx * mx + my * my + mz * mz - r * r
    disc = B * B - 4.0 * C
    if disc < 0.0: return math.inf
    sqrt_disc = math.sqrt(disc)
    t1 = (-B - sqrt_disc) / 2.0; t2 = (-B + sqrt_disc) / 2.0
    ts = [t for t in (t1, t2) if t >= 0.0]
    return min(ts) if ts else math.inf

def simulate_scan_3d(cfg: Sim3DConfig, obstacles: List[Obstacle3D]):
# This function simulates how a LiDAR scanner would emit rays in 3D, checks for intersection and records the hit points(point clouds)

    az_list, el_list = angle_grids(cfg)
    hits = []; origin = (0.0, 0.0, 0.0)
    for el in el_list:
        for az in az_list:
            d = direction_from_angles(az, el)
            nearest_t = cfg.max_range
            for obs in obstacles:
                t = ray_sphere_intersection(origin, d, (obs.x, obs.y, obs.z), obs.radius)
                if t < nearest_t: nearest_t = t
            if nearest_t < cfg.max_range:
                hx = origin[0] + nearest_t * d[0]
                hy = origin[1] + nearest_t * d[1]
                hz = origin[2] + nearest_t * d[2]
                hits.append((round(hx, 3), round(hy, 3), round(hz, 3)))
    return hits

# --------------------------- LABELING FUNCTIONS ---------------------------

def nearest_point_info(point_cloud, cfg: Sim3DConfig):
# It finds the closest detected obstacle point and gives 1. the distance/range from LiDAR
# and 2. the azimuth angle
# Basically the nearest object and at what angle

    if not point_cloud: return cfg.max_range, 0.0
    min_idx = min(range(len(point_cloud)), key=lambda i: (point_cloud[i][0]**2 + point_cloud[i][1]**2 + point_cloud[i][2]**2)) # x^2 + y^2 + z^2
    px, py, pz = point_cloud[min_idx] # Takes the closest point
    dist = (px*px + py*py + pz*pz) ** 0.5
    az_deg = math.degrees(math.atan2(py, px))
    return float(round(dist, 3)), float(round(az_deg, 1))

def classify_rel_velocity(prev_min: float, curr_min: float, eps: float):
# This function classifies the relative motion based on consecutive scans and concludes if the object is static, approaching
# or receding as mentioned in the specifications
# This classifies with a tolerance value of eps as mentioned above
    delta = curr_min - prev_min
    if delta <= -eps: return "approaching"
    if delta >= eps:  return "receding"
    return "static"

def estimate_ttc(prev_min: float, curr_min: float, dt: float):
# This implements Time-To-Collision estimation that is how long before the device collides if we continue at the same speed
    approach_speed = max(0.0, (prev_min - curr_min) / dt)
    if approach_speed <= 1e-6: return math.inf
    return curr_min / approach_speed # Formula to calculate estimate

def label_collision_risk(prev_min: float, curr_min: float, cfg: Sim3DConfig):
# Collision risk classifier, takes the LiDAR's nearest distance measurements from two consecutive scans, uses the TTC estimator
# and decides if it should raise a risk alert

    ttc = estimate_ttc(prev_min, curr_min, cfg.dt)
    if (curr_min < cfg.risk_dist_thresh and (prev_min - curr_min) >= cfg.vel_eps) or (ttc < cfg.ttc_thresh):
        return 1
    return 0

def recommend_maneuver(risk: int, nearest_az_deg: float, nearest_dist: float):
# Takes in the nearest obstacle info and the collision risk flag 0/1 and recommends what the device should do.
    if risk == 1:
        if abs(nearest_az_deg) <= 10 and nearest_dist < 1.0: return "stop"
        return "turn_left" if nearest_az_deg > 0 else "turn_right"
    return "maintain"

# --------------------------- SCENARIO GENERATION ---------------------------

def random_obstacle_3d(cfg: Sim3DConfig):
# This function does the following: 
# - GEnerates a random position within LiDAR FOV and range
# - Converts spherical sampling into cartesian
# - Assigns a random velocity vector using azimuth and elevation direction
# - Random physical radius for obstacle

    # r = random.uniform(1.0, cfg.max_range * 0.9)
    # r = random.uniform(0.5, 10.0)
    # # az = math.radians(random.uniform(-cfg.az_fov_deg/2, cfg.az_fov_deg/2))
    # az = math.radians(random.uniform(-60, 60))  # restrict to ±60° instead of full 180°

    # # el = math.radians(random.uniform(-cfg.el_fov_deg/2, cfg.el_fov_deg/2))
    # el = math.radians(random.uniform(-10, 10))  # obstacles mostly in front, not flying above

# New code to generate stop conditions at least one in 20% of generated obstacles
    p = random.random()
    # Force "stop" obstacles sometimes
    if p < 0.1:   # 20% chance to create a stop scenario
        r = random.uniform(0.5, 1.0)   # very close
        az = math.radians(random.uniform(-10, 10))  # directly in front
        el = math.radians(random.uniform(-5, 5))    # ground level
    
    elif p < 0.2:
    # force turn_left → obstacle on right side
        r = random.uniform(1.0, cfg.max_range * 0.7)
        az = math.radians(random.uniform(20, 60))   # right side
        el = math.radians(random.uniform(-5, 5))

    elif p < 0.3:
    # force turn_right → obstacle on left side
        r = random.uniform(1.0, cfg.max_range * 0.7)
        az = math.radians(random.uniform(-60, -20)) # left side
        el = math.radians(random.uniform(-5, 5))

    else:
        r = random.uniform(0.5, cfg.max_range * 0.9)   # normal distribution
        az = math.radians(random.uniform(-cfg.az_fov_deg/2, cfg.az_fov_deg/2))
        el = math.radians(random.uniform(-cfg.el_fov_deg/2, cfg.el_fov_deg/2))



    x = r * math.cos(el) * math.cos(az)
    y = r * math.cos(el) * math.sin(az)
    z = r * math.sin(el)
    speed = random.uniform(0.0, 2.0)
    v_az = random.uniform(-math.pi, math.pi)
    v_el = random.uniform(-math.pi/6, math.pi/6)
    vx = speed * math.cos(v_el) * math.cos(v_az)
    vy = speed * math.cos(v_el) * math.sin(v_az)
    vz = speed * math.sin(v_el)
    radius = random.uniform(0.2, 0.6)
    return Obstacle3D(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, radius=radius)

def make_scenario_3d(cfg: Sim3DConfig, t_steps: int):
# Generates a random 3D simulation scenario.
# Returns a 3D scenario with generated obstacles and a total number of simulation steps

    n_obs = random.randint(2, 6)
    obs = [random_obstacle_3d(cfg) for _ in range(n_obs)]
    return Scenario3D(obstacles=obs, t_steps=t_steps)

def run_scenario_3d(cfg: Sim3DConfig, scenario: Scenario3D):
# Core simulation loop
# Runs the scenario for scenario.t_steps time steps.
# At each step:
# Scans environment (simulate_scan_3d).
# Finds nearest obstacle.
# Checks motion trend (approaching/receding).
# Assesses collision risk.
# Recommends a maneuver.
# Logs everything.
# Moves obstacles forward in time.

    rows = []; prev_min = None
    for t in range(scenario.t_steps):
        hits = simulate_scan_3d(cfg, scenario.obstacles)
        min_dist, min_az_deg = nearest_point_info(hits, cfg)
        if prev_min is None:
            rel_vel = "static"
            risk = 1 if min_dist < cfg.risk_dist_thresh else 0
        else:
            rel_vel = classify_rel_velocity(prev_min, min_dist, cfg.vel_eps)
            risk = label_collision_risk(prev_min, min_dist, cfg)

            # Normal maneuver recommendation
        maneuver = recommend_maneuver(risk, min_az_deg, min_dist)

        # Bias increase the chance of exception cases-- 20% each. Makes 
        # sure stop/turn_left/turn_right occur more frequently.
        
        if maneuver == "maintain":
            prob = random.random()
            if prob < 0.067:
                maneuver = "stop"
            elif prob < 0.134:
                maneuver = "turn_left"
            elif prob < 0.201:
                maneuver = "turn_right"
            # else keep maintain
        row = {
            "lidar_scan_data": json.dumps(hits),
            "obstacle_distance": min_dist,
            "obstacle_angle": min_az_deg,
            "relative_velocity": rel_vel,
            "collision_risk": int(risk),
            "recommended_maneuver": maneuver
        }
        rows.append(row)
        prev_min = min_dist
        for obs in scenario.obstacles: obs.step(cfg.dt)
    return rows

# --------------------------- MAIN FUNCTION ---------------------------

# Driver function
# generate_and_save creates multiple 3D driving scenarios, runs them through LiDAR simulation, collects results, shuffles them,
# and saves the train and test datasets train.csv and test.csv
def generate_and_save(out_dir='data3d', n_train=800, n_test=200, t_steps=12, seed=11):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    cfg = Sim3DConfig()
    train_rows = []; test_rows = []
    for _ in range(n_train):
        sc = make_scenario_3d(cfg, t_steps)
        train_rows.extend(run_scenario_3d(cfg, sc))
    for _ in range(n_test):
        sc = make_scenario_3d(cfg, t_steps)
        test_rows.extend(run_scenario_3d(cfg, sc))
    df_train = pd.DataFrame(train_rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_test = pd.DataFrame(test_rows).sample(frac=1.0, random_state=43).reset_index(drop=True)
    train_path = os.path.join(out_dir, 'train.csv')
    test_path = os.path.join(out_dir, 'test.csv')
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    return train_path, test_path, len(df_train), len(df_test)

if __name__ == '__main__':
    print(generate_and_save())
