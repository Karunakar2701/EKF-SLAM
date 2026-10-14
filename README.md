# EKF-SLAM Simulation

A Python-based 2D mobile robot simulation demonstrating **Extended Kalman Filter Simultaneous Localization and Mapping (EKF-SLAM)**. The robot estimates its pose while building a map of landmarks from noisy range-and-bearing observations.

This project is a simulation for learning and experimentation. It does **not** currently use ROS 2 or Gazebo.

## Features

- Unicycle-style 2D robot motion model
- Simulated motion disturbances and noisy odometry
- Range-and-bearing sensor model
- Configurable sensor range and field of view
- Probabilistic missed detections and Gaussian measurement noise
- Unordered observations without landmark IDs
- Mahalanobis-distance nearest-neighbor data association with gating
- Dynamic state vector and covariance matrix as landmarks are added
- Landmark initialization using the inverse measurement model
- Robot–landmark cross-covariance propagation
- Joseph-form EKF covariance update
- Live visualization of the robot, estimated trajectory, and landmarks
- Mean absolute error (MAE) reporting for the robot pose
- Multi-seed robustness testing

## How It Works

The EKF-SLAM state is represented as:

```text
X = [x, y, yaw, lx0, ly0, lx1, ly1, ...]
```

where `(x, y, yaw)` is the robot pose and each `(lx, ly)` pair represents a mapped landmark. The covariance matrix `P` represents uncertainty across the joint robot-and-landmark state.

Each simulation step broadly follows this pipeline:

1. Generate commanded robot motion.
2. Simulate actual motion with disturbances and produce noisy odometry.
3. Predict the robot state and covariance using the EKF motion model.
4. Generate noisy range-and-bearing observations for visible landmarks.
5. Associate each observation with an existing landmark using Mahalanobis distance and a gate.
6. Update the state for a matched landmark, or initialize a new landmark when no match passes the gate.
7. Update the visualization and track pose-estimation error.

## Requirements

- Python 3.9 or newer recommended
- NumPy
- Matplotlib


## Run the Simulation

Save the Python source file in the repository (for example, as `EKF_SLAM.py`), then run:

```bash
python ekf_slam_simulation.py
```

The default entry point runs the live simulation with seed `42` and real-time plotting enabled.

## Configuration

The main simulation settings are defined near the top of the Python file:

| Setting | Default | Description |
|---|---:|---|
| `DT` | `0.10` s | Simulation time step |
| `SIM_TIME` | `35.0` s | Total simulation duration |
| `MAX_RANGE` | `7.0` m | Maximum sensor range |
| `MIN_RANGE` | `0.25` m | Minimum sensor range |
| `FOV` | `100°` | Total sensor field of view |
| `DETECTION_PROBABILITY` | `0.98` | Probability of detecting a visible landmark |
| `SIGMA_MOTION_V` | `0.03` m/s | Translational motion disturbance standard deviation |
| `SIGMA_MOTION_W` | `0.01` rad/s | Rotational motion disturbance standard deviation |
| `SIGMA_ODOM_V` | `0.03` m/s | Odometry velocity noise standard deviation |
| `SIGMA_ODOM_W` | `0.01` rad/s | Odometry yaw-rate noise standard deviation |
| `SIGMA_RANGE` | `0.05` m | Range measurement noise standard deviation |
| `SIGMA_BEARING` | `1°` | Bearing measurement noise standard deviation |
| `GATE_THRESHOLD` | `13.82` | Squared Mahalanobis-distance association threshold |

Edit these constants to experiment with different sensor and motion conditions.

## Project Structure

A simple repository layout could be:

```text
EKF-SLAM/
├── EKF_SLAM.py
├── README.md
└── requirements.txt
```

Example `requirements.txt`:

```text
numpy
matplotlib
```

## Limitations

- This is a simulation, not a deployment-ready robotics stack.
- ROS 2, Gazebo, and physical robot hardware are not integrated.
- The environment uses a predefined set of landmarks.
- Data association uses nearest-neighbor Mahalanobis gating; incorrect associations may still occur.
- Performance depends on noise settings, landmark geometry, and random seed.
- Results from a single run do not establish robustness; use multiple seeds when evaluating behavior.

## Future Improvements

- Integrate with ROS 2 odometry and sensor topics.
- Add a Gazebo simulation environment.
- Compare estimated landmarks and trajectories against ground truth quantitatively.
- Add automated tests for Jacobians, covariance properties, and data association.
- Export plots and experiment metrics for reproducible comparisons.

## Concepts Demonstrated

- Extended Kalman Filter (EKF)
- Simultaneous Localization and Mapping (SLAM)
- Robot motion and measurement models
- Jacobians and covariance propagation
- Probabilistic sensor simulation
- Landmark initialization and data association
- State estimation and uncertainty
