
"""
REALISTIC(ISH) EKF-SLAM SIMULATION
===================================

This is a self-contained Python EKF-SLAM simulator.

It is intentionally closer to a real robot than the tiny
textbook demo, while still being simple enough to study.

SIMULATED SYSTEM
----------------
True robot:
    unicycle / differential-drive style motion

Odometry:
    noisy velocity + noisy yaw-rate measurement

Sensor:
    range + bearing measurements
    finite range
    finite field of view
    missed detections
    Gaussian range/bearing noise
    observations arrive in random order

EKF-SLAM:
    unknown landmark IDs
    Mahalanobis nearest-neighbor association
    gated association
    dynamic state vector
    dynamic covariance matrix
    landmark initialization from inverse measurement model
    robot <-> landmark cross covariance
    Joseph-form covariance update

STATE
-----
X = [x, y, yaw,
     lx0, ly0,
     lx1, ly1,
     ...]

P is the covariance of the complete joint state.

IMPORTANT
---------
This is a realistic-ish SIMULATION, not ROS2/Gazebo yet.
The next stage can replace the simulated odometry/sensor with
ROS2 topics such as /odom and /scan.

"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. SIMULATION SETTINGS
# ============================================================

DT = 0.10
SIM_TIME = 35.0


# ============================================================
# 2. SENSOR MODEL
# ============================================================

# Maximum and minimum sensing range.
MAX_RANGE = 7.0
MIN_RANGE = 0.25

# Sensor field of view:
# +/- 50 degrees around the robot heading.
FOV = np.deg2rad(100.0)

# Probability that a visible landmark is detected.
DETECTION_PROBABILITY = 0.98


# ============================================================
# 3. TRUE ROBOT MOTION NOISE
# ============================================================

# What the physical robot actually does.
#
# These represent small motion disturbances/slip.
SIGMA_MOTION_V = 0.03       # m/s
SIGMA_MOTION_W = 0.01       # rad/s


# ============================================================
# 4. ODOMETRY NOISE
# ============================================================

# What the wheel/odometry system reports to the EKF.
#
# The EKF does NOT receive the true velocity.
SIGMA_ODOM_V = 0.03         # m/s
SIGMA_ODOM_W = 0.01         # rad/s


# ============================================================
# 5. SENSOR NOISE
# ============================================================

SIGMA_RANGE = 0.05          # m
SIGMA_BEARING = np.deg2rad(1.0)  # rad

R = np.diag([
    SIGMA_RANGE ** 2,
    SIGMA_BEARING ** 2,
])


# ============================================================
# 6. ODOMETRY PROCESS NOISE
# ============================================================

Q_U = np.diag([
    SIGMA_ODOM_V ** 2,
    SIGMA_ODOM_W ** 2,
])


# ============================================================
# 7. INITIAL ROBOT COVARIANCE
# ============================================================

INITIAL_P = np.diag([
    0.5 ** 2,
    0.5 ** 2,
    np.deg2rad(10.0) ** 2,
])


# ============================================================
# 8. DATA ASSOCIATION
# ============================================================

# Squared Mahalanobis distance gate for a 2D measurement.
#
# This is intentionally permissive because the map is built
# from noisy observations and the EKF is not globally perfect.
GATE_THRESHOLD = 13.82


# ============================================================
# 9. STATE CONSTANTS
# ============================================================

STATE_SIZE = 3
LM_SIZE = 2


# ============================================================
# 10. WORLD MAP
# ============================================================

# Fixed landmarks in the world.
#
# They are deliberately separated enough that the sensor
# can usually distinguish neighboring landmarks.
TRUE_LANDMARKS = np.array([
    [4.0,  3.0],
    [9.0,  8.0],
    [14.0, 3.0],
    [18.0, 10.0],
    [13.0, 17.0],
    [6.0, 16.0],
    [0.0, 11.0],
    [-3.0, 6.0],
], dtype=float)


# ============================================================
# 11. BASIC UTILITIES
# ============================================================

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    """
    return (
        (angle + np.pi)
        % (2.0 * np.pi)
    ) - np.pi


def number_of_landmarks(state):
    """
    Number of landmarks currently inside the SLAM state.
    """
    return (
        len(state) - STATE_SIZE
    ) // LM_SIZE


# ============================================================
# 12. ROBOT MOTION MODEL
# ============================================================

def motion_model(state, control, dt):
    """
    Unicycle motion model.

    state:
        [x, y, yaw, ...]

    control:
        [v, w]

    Landmarks stay fixed.
    """
    new_state = state.copy()

    x, y, yaw = state[:3]
    v, w = control

    new_state[0] = (
        x + v * np.cos(yaw) * dt
    )

    new_state[1] = (
        y + v * np.sin(yaw) * dt
    )

    new_state[2] = normalize_angle(
        yaw + w * dt
    )

    return new_state


# ============================================================
# 13. MOTION JACOBIAN F = df/dx
# ============================================================

def motion_jacobian(state, control, dt):
    """
    Full-state motion Jacobian.

    Landmark rows/columns remain identity because landmarks
    are static.
    """
    yaw = state[2]
    v = control[0]

    n = len(state)

    F = np.eye(n)

    F[0, 2] = (
        -v * np.sin(yaw) * dt
    )

    F[1, 2] = (
        v * np.cos(yaw) * dt
    )

    return F


# ============================================================
# 14. CONTROL JACOBIAN G = df/du
# ============================================================

def control_jacobian(state, dt):
    """
    Maps uncertainty in [v, w] into [x, y, yaw].
    """
    yaw = state[2]

    return np.array([
        [
            np.cos(yaw) * dt,
            0.0,
        ],
        [
            np.sin(yaw) * dt,
            0.0,
        ],
        [
            0.0,
            dt,
        ],
    ])


# ============================================================
# 15. FULL PROCESS NOISE
# ============================================================

def full_process_noise(state, dt):
    """
    Q_full = G_u Q_u G_u^T

    Only the robot state receives direct process noise.
    Landmarks are static.
    """
    n = len(state)

    Q_full = np.zeros(
        (n, n)
    )

    G_u = control_jacobian(
        state,
        dt
    )

    Q_robot = (
        G_u
        @ Q_U
        @ G_u.T
    )

    Q_full[
        :STATE_SIZE,
        :STATE_SIZE
    ] = Q_robot

    return Q_full


# ============================================================
# 16. EKF PREDICTION
# ============================================================

def predict(
    state,
    P,
    odometry,
    dt
):
    """
    Predict using noisy odometry.
    """
    F = motion_jacobian(
        state,
        odometry,
        dt
    )

    predicted_state = motion_model(
        state,
        odometry,
        dt
    )

    predicted_P = (
        F @ P @ F.T
        + full_process_noise(
            state,
            dt
        )
    )

    predicted_P = 0.5 * (
        predicted_P
        + predicted_P.T
    )

    return (
        predicted_state,
        predicted_P
    )


# ============================================================
# 17. PREDICTED SENSOR MEASUREMENT
# ============================================================

def measurement_model(
    state,
    landmark_index
):
    """
    Predict [range, bearing] to one landmark.
    """
    x, y, yaw = state[:3]

    lm_start = (
        STATE_SIZE
        + LM_SIZE * landmark_index
    )

    lx = state[lm_start]
    ly = state[lm_start + 1]

    dx = lx - x
    dy = ly - y

    r = math.hypot(
        dx,
        dy
    )

    if r < 1e-12:
        raise ValueError(
            "Robot too close to landmark."
        )

    bearing = normalize_angle(
        math.atan2(dy, dx)
        - yaw
    )

    return np.array([
        r,
        bearing,
    ])


# ============================================================
# 18. MEASUREMENT JACOBIAN H
# ============================================================

def measurement_jacobian(
    state,
    landmark_index
):
    """
    Full 2 x N measurement Jacobian.
    """
    x, y, yaw = state[:3]

    lm_start = (
        STATE_SIZE
        + LM_SIZE * landmark_index
    )

    lx = state[lm_start]
    ly = state[lm_start + 1]

    dx = lx - x
    dy = ly - y

    q = (
        dx * dx
        + dy * dy
    )

    if q < 1e-12:
        raise ValueError(
            "Robot too close to landmark."
        )

    r = math.sqrt(q)

    H = np.zeros(
        (2, len(state))
    )

    # Robot part
    H[0, 0] = -dx / r
    H[0, 1] = -dy / r

    H[1, 0] = dy / q
    H[1, 1] = -dx / q
    H[1, 2] = -1.0

    # Landmark part
    H[0, lm_start] = dx / r
    H[0, lm_start + 1] = dy / r

    H[1, lm_start] = -dy / q
    H[1, lm_start + 1] = dx / q

    return H


# ============================================================
# 19. TRUE SENSOR MODEL
# ============================================================

def true_measurement(
    true_state,
    landmark
):
    """
    Ideal range/bearing measurement from the true robot
    to a true landmark.
    """
    x, y, yaw = true_state

    lx, ly = landmark

    dx = lx - x
    dy = ly - y

    r = math.hypot(
        dx,
        dy
    )

    bearing = normalize_angle(
        math.atan2(dy, dx)
        - yaw
    )

    return r, bearing


# ============================================================
# 20. NEW LANDMARK INITIALIZATION
# ============================================================

def initialize_landmark(
    state,
    P,
    measurement
):
    """
    Add a new landmark using the inverse sensor model.

    The new landmark covariance includes:

        robot uncertainty
        +
        sensor uncertainty
        +
        robot <-> landmark cross covariance
    """
    r, bearing = measurement

    x, y, yaw = state[:3]

    alpha = yaw + bearing

    lx = (
        x
        + r * np.cos(alpha)
    )

    ly = (
        y
        + r * np.sin(alpha)
    )

    new_landmark = np.array([
        lx,
        ly,
    ])

    # Jacobian wrt robot pose
    Gx = np.array([
        [
            1.0,
            0.0,
            -r * np.sin(alpha),
        ],
        [
            0.0,
            1.0,
             r * np.cos(alpha),
        ],
    ])

    # Jacobian wrt measurement
    Gz = np.array([
        [
            np.cos(alpha),
            -r * np.sin(alpha),
        ],
        [
            np.sin(alpha),
             r * np.cos(alpha),
        ],
    ])

    P_x_robot = P[
        :,
        :STATE_SIZE
    ]

    P_robot = P[
        :STATE_SIZE,
        :STATE_SIZE
    ]

    # Existing state <-> new landmark
    P_xm = (
        P_x_robot
        @ Gx.T
    )

    # New landmark covariance
    P_mm = (
        Gx
        @ P_robot
        @ Gx.T
        +
        Gz
        @ R
        @ Gz.T
    )

    old_size = len(state)

    new_state = np.concatenate([
        state,
        new_landmark,
    ])

    new_P = np.zeros(
        (
            old_size + LM_SIZE,
            old_size + LM_SIZE,
        )
    )

    # Copy old covariance
    new_P[
        :old_size,
        :old_size
    ] = P

    # Existing -> new landmark
    new_P[
        :old_size,
        old_size:
    ] = P_xm

    # New landmark -> existing
    new_P[
        old_size:,
        :old_size
    ] = P_xm.T

    # New landmark covariance
    new_P[
        old_size:,
        old_size:
    ] = P_mm

    new_P = 0.5 * (
        new_P
        + new_P.T
    )

    return (
        new_state,
        new_P
    )


# ============================================================
# 21. MAHALANOBIS DATA ASSOCIATION
# ============================================================

def associate_measurement(
    state,
    P,
    measurement,
):
    """
    Nearest-neighbor Mahalanobis association.

    Returns:
        best landmark index, best d2
        or
        None, best d2 if the gate rejects all landmarks
    """
    n_lm = number_of_landmarks(
        state
    )

    if n_lm == 0:
        return None, None

    best_index = None
    best_d2 = float("inf")

    for i in range(n_lm):

        z_pred = (
            measurement_model(
                state,
                i
            )
        )

        innovation = (
            measurement
            - z_pred
        )

        innovation[1] = (
            normalize_angle(
                innovation[1]
            )
        )

        H = (
            measurement_jacobian(
                state,
                i
            )
        )

        S = (
            H @ P @ H.T
            + R
        )

        d2 = float(
            innovation.T
            @ np.linalg.solve(
                S,
                innovation
            )
        )

        if d2 < best_d2:
            best_d2 = d2
            best_index = i

    if best_d2 <= GATE_THRESHOLD:
        return best_index, best_d2

    return None, best_d2


# ============================================================
# 22. JOSEPH-FORM EKF UPDATE
# ============================================================

def update_landmark(
    state,
    P,
    measurement,
    landmark_index
):
    """
    EKF correction for an existing landmark.
    """
    z_pred = (
        measurement_model(
            state,
            landmark_index
        )
    )

    innovation = (
        measurement
        - z_pred
    )

    innovation[1] = (
        normalize_angle(
            innovation[1]
        )
    )

    H = (
        measurement_jacobian(
            state,
            landmark_index
        )
    )

    S = (
        H @ P @ H.T
        + R
    )

    K = (
        P
        @ H.T
        @ np.linalg.solve(
            S,
            np.eye(2)
        )
    )

    new_state = (
        state
        + K @ innovation
    )

    new_state[2] = (
        normalize_angle(
            new_state[2]
        )
    )

    I = np.eye(
        len(state)
    )

    A = I - K @ H

    new_P = (
        A @ P @ A.T
        + K @ R @ K.T
    )

    new_P = 0.5 * (
        new_P
        + new_P.T
    )

    return (
        new_state,
        new_P
    )


# ============================================================
# 23. PROCESS ONE OBSERVATION
# ============================================================

def process_measurement(
    state,
    P,
    measurement,
):
    """
    Data-associate the measurement.

    If matched:
        EKF update.

    If unmatched:
        create a new landmark.
    """
    landmark_index, d2 = (
        associate_measurement(
            state,
            P,
            measurement
        )
    )

    if landmark_index is None:

        new_state, new_P = (
            initialize_landmark(
                state,
                P,
                measurement
            )
        )

        return (
            new_state,
            new_P,
            "new",
            d2
        )

    new_state, new_P = (
        update_landmark(
            state,
            P,
            measurement,
            landmark_index
        )
    )

    return (
        new_state,
        new_P,
        "matched",
        d2
    )


# ============================================================
# 24. COMMANDED ROBOT MOTION
# ============================================================

def commanded_control(step):
    """
    Smooth indoor-robot trajectory.

    The robot drives forward while slowly varying
    its yaw rate.
    """
    v = 0.7

    w = (
        0.35
        * math.sin(
            step * 0.04
        )
    )

    return np.array([
        v,
        w,
    ])


# ============================================================
# 25. TRUE ROBOT + ODOMETRY SIMULATION
# ============================================================

def simulate_robot_and_odometry(
    true_state,
    commanded_control,
    rng,
):
    """
    Simulate:

        commanded velocity
              |
              v
        actual robot motion
              |
              v
        noisy odometry reading

    The EKF receives odometry, not ground truth.
    """
    actual_control = (
        commanded_control
        + rng.normal(
            0.0,
            [
                SIGMA_MOTION_V,
                SIGMA_MOTION_W,
            ],
        )
    )

    new_true_state = (
        motion_model(
            true_state,
            actual_control,
            DT
        )
    )

    odometry = (
        actual_control
        + rng.normal(
            0.0,
            [
                SIGMA_ODOM_V,
                SIGMA_ODOM_W,
            ],
        )
    )

    return (
        new_true_state,
        odometry,
    )


# ============================================================
# 26. SENSOR SIMULATION
# ============================================================

def simulate_sensor(
    true_state,
    rng,
):
    """
    Create an unordered set of noisy range/bearing observations.

    The true landmark index is NEVER passed to EKF-SLAM.
    """
    observations = []

    for landmark in TRUE_LANDMARKS:

        r_true, bearing_true = (
            true_measurement(
                true_state,
                landmark
            )
        )

        # Range gate
        if not (
            MIN_RANGE
            <= r_true
            <= MAX_RANGE
        ):
            continue

        # Field of view
        if abs(bearing_true) > FOV:
            continue

        # Missed detection
        if (
            rng.random()
            > DETECTION_PROBABILITY
        ):
            continue

        noisy_range = (
            r_true
            + rng.normal(
                0.0,
                SIGMA_RANGE
            )
        )

        noisy_bearing = (
            bearing_true
            + rng.normal(
                0.0,
                SIGMA_BEARING
            )
        )

        if noisy_range < MIN_RANGE:
            continue

        noisy_bearing = normalize_angle(
            noisy_bearing
        )

        observations.append(
            np.array([
                noisy_range,
                noisy_bearing,
            ])
        )

    # Real sensor output does not tell EKF
    # which landmark each observation belongs to.
    rng.shuffle(
        observations
    )

    return observations


# ============================================================
# 27. RUN ONE SIMULATION
# ============================================================

def run_simulation(
    seed=0,
    return_full_state=False,
):
    """
    Run one simulation with a fresh random generator.
    """
    rng = np.random.default_rng(
        seed
    )

    state = np.array([
        0.0,
        0.0,
        0.0,
    ])

    P = INITIAL_P.copy()

    true_state = np.array([
        0.0,
        0.0,
        0.0,
    ])

    true_history = []
    estimated_history = []

    steps = int(
        35.0 / DT
    )

    new_count = 0
    match_count = 0
    ignored_count = 0

    for step in range(steps):

        commanded = (
            commanded_control(
                step
            )
        )

        # True robot motion + noisy odometry
        (
            true_state,
            odometry,
        ) = simulate_robot_and_odometry(
            true_state,
            commanded,
            rng,
        )

        # Prediction from odometry
        state, P = predict(
            state,
            P,
            odometry,
            DT,
        )

        # Unordered sensor observations
        observations = (
            simulate_sensor(
                true_state,
                rng,
            )
        )

        # Process each observation
        for measurement in observations:

            (
                state,
                P,
                result,
                d2,
            ) = process_measurement(
                state,
                P,
                measurement,
            )

            if result == "new":
                new_count += 1
            else:
                match_count += 1

        true_history.append(
            true_state.copy()
        )

        estimated_history.append(
            state[:3].copy()
        )

    true_history = np.array(
        true_history
    )

    estimated_history = np.array(
        estimated_history
    )

    error = (
        true_history
        - estimated_history
    )

    error[:, 2] = np.array([
        normalize_angle(v)
        for v in error[:, 2]
    ])

    mae = np.mean(
        np.abs(error),
        axis=0
    )

    result = {
        "seed": seed,
        "mae": mae,
        "true_state": true_state.copy(),
        "estimated_state": (
            state[:3].copy()
        ),
        "state": state.copy(),
        "P": P.copy(),
        "true_history": true_history,
        "estimated_history": estimated_history,
        "num_landmarks": (
            number_of_landmarks(state)
        ),
        "new_landmarks": new_count,
        "matched_measurements": match_count,
    }

    if not return_full_state:
        # Keep the same information; the full state is already
        # cheap for this small simulator.
        return result

    return result


# ============================================================
# 28. ROBUSTNESS TEST
# ============================================================

def robustness_test(
    seeds=range(20)
):
    """
    Run many seeds.

    This is important because one lucky random seed is not
    enough evidence that a SLAM implementation works.
    """
    results = []

    for seed in seeds:

        result = run_simulation(
            seed
        )

        results.append(result)

    print("\n" + "=" * 72)
    print("20-SEED ROBUSTNESS TEST")
    print("=" * 72)

    print(
        "seed | x_MAE | y_MAE | "
        "theta_MAE | landmarks"
    )

    for result in results:

        mae = result["mae"]

        print(
            f"{result['seed']:4d} | "
            f"{mae[0]:6.3f} | "
            f"{mae[1]:6.3f} | "
            f"{mae[2]:9.3f} | "
            f"{result['num_landmarks']:9d}"
        )

    x_mae = np.array([
        r["mae"][0]
        for r in results
    ])

    y_mae = np.array([
        r["mae"][1]
        for r in results
    ])

    theta_mae = np.array([
        r["mae"][2]
        for r in results
    ])

    print("\nAVERAGE")
    print(
        f"x     : {x_mae.mean():.3f} "
        f"+/- {x_mae.std():.3f} m"
    )

    print(
        f"y     : {y_mae.mean():.3f} "
        f"+/- {y_mae.std():.3f} m"
    )

    print(
        f"theta : {theta_mae.mean():.3f} "
        f"+/- {theta_mae.std():.3f} rad"
    )

    print("\nWORST")
    print(
        f"x     : {x_mae.max():.3f} m"
    )

    print(
        f"y     : {y_mae.max():.3f} m"
    )

    print(
        f"theta : {theta_mae.max():.3f} rad"
    )

    return results


# ============================================================
# 29. PLOT ONE FINAL RUN
# ============================================================

def plot_run(result):
    """
    Plot true/estimated robot trajectory and map.
    """
    plt.figure(
        figsize=(11, 8)
    )

    true_history = result[
        "true_history"
    ]

    estimated_history = result[
        "estimated_history"
    ]

    plt.plot(
        true_history[:, 0],
        true_history[:, 1],
        label="True robot",
        linewidth=2,
    )

    plt.plot(
        estimated_history[:, 0],
        estimated_history[:, 1],
        "--",
        label="EKF-SLAM robot",
        linewidth=2,
    )

    # True map
    plt.scatter(
        TRUE_LANDMARKS[:, 0],
        TRUE_LANDMARKS[:, 1],
        marker="x",
        s=90,
        label="True landmarks",
    )

    # Estimated map
    state = result["state"]

    n_lm = number_of_landmarks(
        state
    )

    for i in range(n_lm):

        start = (
            STATE_SIZE
            + LM_SIZE * i
        )

        lx = state[start]
        ly = state[start + 1]

        plt.scatter(
            lx,
            ly,
            marker="o",
            facecolors="none",
            s=100,
        )

        plt.text(
            lx + 0.10,
            ly + 0.10,
            f"L{i}",
        )

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    plt.title(
        "EKF-SLAM: realistic 2D simulation"
    )

    plt.axis(
        "equal"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.show()



# ============================================================
# 30. LIVE / REAL-TIME SIMULATION
# ============================================================

def run_live_simulation(seed=42, realtime=True):
    """
    Run EKF-SLAM and visualize it continuously.

    The figure is updated once per simulation step, so it behaves
    like a live 2-D robot simulation.

    realtime=True:
        pause for DT seconds between frames.

    realtime=False:
        run as fast as matplotlib can update the display.
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Reset simulation
    # --------------------------------------------------------

    state = np.array([
        0.0,
        0.0,
        0.0,
    ])

    P = INITIAL_P.copy()

    true_state = np.array([
        0.0,
        0.0,
        0.0,
    ])

    true_history = []
    estimated_history = []

    steps = int(
        SIM_TIME / DT
    )

    # --------------------------------------------------------
    # Create figure once.
    # --------------------------------------------------------

    plt.ion()

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    for step in range(steps):

        # ====================================================
        # TRUE ROBOT + ODOMETRY
        # ====================================================

        commanded = commanded_control(
            step
        )

        true_state, odometry = (
            simulate_robot_and_odometry(
                true_state,
                commanded,
                rng,
            )
        )

        # ====================================================
        # EKF PREDICTION
        # ====================================================

        state, P = predict(
            state,
            P,
            odometry,
            DT,
        )

        # ====================================================
        # SENSOR
        # ====================================================

        observations = simulate_sensor(
            true_state,
            rng,
        )

        # ====================================================
        # EKF MEASUREMENT UPDATES
        # ====================================================

        new_count = 0
        matched_count = 0

        for measurement in observations:

            (
                state,
                P,
                result,
                _,
            ) = process_measurement(
                state,
                P,
                measurement,
            )

            if result == "new":
                new_count += 1
            else:
                matched_count += 1

        # ====================================================
        # SAVE HISTORY
        # ====================================================

        true_history.append(
            true_state.copy()
        )

        estimated_history.append(
            state[:3].copy()
        )

        # ====================================================
        # REDRAW FRAME
        # ====================================================

        ax.clear()

        # True landmarks
        ax.scatter(
            TRUE_LANDMARKS[:, 0],
            TRUE_LANDMARKS[:, 1],
            marker="x",
            s=90,
            label="True landmarks",
        )

        # Estimated landmarks
        n_lm = number_of_landmarks(
            state
        )

        for i in range(n_lm):

            start = (
                STATE_SIZE
                + LM_SIZE * i
            )

            lx = state[start]
            ly = state[start + 1]

            ax.scatter(
                lx,
                ly,
                marker="o",
                facecolors="none",
                s=100,
            )

            ax.text(
                lx + 0.12,
                ly + 0.12,
                f"L{i}",
                fontsize=8,
            )

        # True trajectory
        true_hist = np.array(
            true_history
        )

        ax.plot(
            true_hist[:, 0],
            true_hist[:, 1],
            linewidth=2,
            label="True robot",
        )

        # Estimated trajectory
        est_hist = np.array(
            estimated_history
        )

        ax.plot(
            est_hist[:, 0],
            est_hist[:, 1],
            "--",
            linewidth=2,
            label="EKF-SLAM robot",
        )

        # ----------------------------------------------------
        # Draw true robot.
        # ----------------------------------------------------

        ax.scatter(
            true_state[0],
            true_state[1],
            marker="o",
            s=90,
            label="True pose",
        )

        ax.arrow(
            true_state[0],
            true_state[1],
            0.8 * np.cos(true_state[2]),
            0.8 * np.sin(true_state[2]),
            head_width=0.18,
            length_includes_head=True,
        )

        # ----------------------------------------------------
        # Draw estimated robot.
        # ----------------------------------------------------

        ax.scatter(
            state[0],
            state[1],
            marker="s",
            s=90,
            label="Estimated pose",
        )

        ax.arrow(
            state[0],
            state[1],
            0.8 * np.cos(state[2]),
            0.8 * np.sin(state[2]),
            head_width=0.18,
            length_includes_head=True,
        )

        # ----------------------------------------------------
        # Draw sensor observations around the estimated robot.
        # ----------------------------------------------------

        for measurement in observations:

            measured_range = measurement[0]
            measured_bearing = measurement[1]

            angle = (
                state[2]
                + measured_bearing
            )

            end_x = (
                state[0]
                + measured_range
                * np.cos(angle)
            )

            end_y = (
                state[1]
                + measured_range
                * np.sin(angle)
            )

            ax.plot(
                [state[0], end_x],
                [state[1], end_y],
                alpha=0.08,
            )

        # ----------------------------------------------------
        # Information panel.
        # ----------------------------------------------------

        ax.set_title(
            "EKF-SLAM live simulation\n"
            f"Step {step + 1}/{steps}  |  "
            f"Time {(step + 1) * DT:.1f} s  |  "
            f"Landmarks: {n_lm}"
        )

        ax.text(
            0.02,
            0.98,
            (
                f"True pose: "
                f"[{true_state[0]:.2f}, "
                f"{true_state[1]:.2f}, "
                f"{true_state[2]:.2f}]\n"
                f"Estimated pose: "
                f"[{state[0]:.2f}, "
                f"{state[1]:.2f}, "
                f"{state[2]:.2f}]\n"
                f"Visible measurements: {len(observations)}\n"
                f"Matched: {matched_count}   "
                f"New: {new_count}\n"
                f"P shape: {P.shape}"
            ),
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                alpha=0.85,
            ),
        )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
        ax.axis("equal")
        ax.legend(
            loc="lower right"
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

        if realtime:
            plt.pause(DT)
        else:
            plt.pause(0.001)

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    plt.ioff()

    true_history = np.array(
        true_history
    )

    estimated_history = np.array(
        estimated_history
    )

    error = (
        true_history
        - estimated_history
    )

    error[:, 2] = np.array([
        normalize_angle(v)
        for v in error[:, 2]
    ])

    mae = np.mean(
        np.abs(error),
        axis=0
    )

    print("\n" + "=" * 70)
    print("LIVE SIMULATION COMPLETE")
    print("=" * 70)

    print(
        "True robot state:"
    )
    print(
        true_state
    )

    print(
        "\nEstimated robot state:"
    )
    print(
        state[:3]
    )

    print(
        "\nEstimated landmarks:"
    )
    print(
        n_lm
    )

    print(
        "\nFinal P shape:"
    )
    print(
        P.shape
    )

    print(
        "\nMean absolute error:"
    )

    print(
        f"x     : {mae[0]:.4f} m"
    )

    print(
        f"y     : {mae[1]:.4f} m"
    )

    print(
        f"theta : {mae[2]:.4f} rad"
    )

    plt.show()

    return {
        "true_state": true_state,
        "estimated_state": state[:3].copy(),
        "num_landmarks": n_lm,
        "P": P,
        "mae": mae,
        "true_history": true_history,
        "estimated_history": estimated_history,
    }


# ============================================================
# 31. MAIN
# ============================================================

if __name__ == "__main__":

    # Set True to make the simulation play in real time.
    # Each EKF step corresponds approximately to DT seconds.
    run_live_simulation(
        seed=42,
        realtime=True,
    )