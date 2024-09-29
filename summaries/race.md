

Organized review of the paper `Autonomous Drone Racing: A Survey, IEEE TRANSACTIONS ON ROBOTICS, VOL. 40, 2024`, with focus on modern guidance and control for agile quadrotors.


## III - B: Classic GNC, Guidance (Planning)

## III - C: Classic GNC, Control, Model-based control

*title:* Geometric tracking control of a quadrotor UAV for extreme maneuverability

Model-based classic control (3-C-1-126) / Geometric tracking control is introduced on the special Euclidean group SE(3) and completely avoids singularities commonly associated with Euler angle formulations on SO(3). This nonlinear controller showed the ability to execute acrobatic maneuvers in simulation and was the first to demonstrate recovery from an inverted initial attitude.

[code](https://github.com/fdcl-gwu/uav_geometric_control)

-----

*title:* Trajectory generation and control for precise aggressive maneuvers with quadrotors

Model-based classic control (3-C-1-112) / Model-based classic control (3-C-1-112) / The dynamic model of a quadrotor is shown to be differentially flatwhen choosing its position and heading as flat outputs in [112]. In this work, many agile maneuvers are performed onboard real drones with speeds up to 2.6 m/s. 

-----

**STAR**

*title:* Differential flatness of quadrotor dynamics subject to rotor drag for accurate tracking of high-speed trajectories

Model-based classic control (3-C-1-26) / proving that the dynamics model of a quadrotor subject to linear rotor drag is also differentially flat. The inclusion of the aerodynamic model within the nonlinear controller led to demonstrated flight speeds up to 4 m/s while reducing tracking error by 50% onboard a real drone.

[code](https://github.com/uzh-rpg/rpg_quadrotor_control)

-----

*title:* Accurate tracking of aggressive quadrotor trajectories using incremental nonlinear dynamic inversion and differential flatness

Model-based classic control (3-C-1-127) / The differential flatness method is further extended in [127] by cascading an incremental nonlinear dynamic inversion (INDI) controller with the differential flatness controller described in [112] but neglects the aerodynamic model addition from [26]. The INDI controller is designed to track the angular acceleration commands Ω̇ from the given reference trajectory. Top speeds of nearly 13 m/s and accelerations over 2 g are demonstrated onboard a real quadrotor. The controller shows robustness against large aerodynamic disturbances in part due to the INDI controller.

-----

*title:* A comparative study of nonlinear MPC and differential-flatness-based control for quadrotor agile flight

Model-based classic control (3-C-1-125) / An investigation of the performance of nonlinear model predictive control (NMPC) against differential flatness methods is available in [125]. Cascaded controllers of INDI-NMPC and INDI-differential flatness are shown to track aggressive racing trajectories that achieve speeds of around 20m/s and accelerations of over 4 g. While differential flatness methods are computationally efficient controllers and relatively easy to implement, they are outperformed on racing tasks by NMPC.

-----

**STAR**

*title:* Data-Driven MPC for Quadrotors

Model-based classic control (3-C-1-40) / Nonlinear MPC methods are also used in [40] where a nominal quadrotor model is augmented with a data-driven model composed of Gaussian processes and used directly within the MPC formulation. The authors found that the Gaussian-process model could capture highly nonlinear aerodynamic behavior, which is difficult to model in practice as described in Section II. The additional terms introduced by the Gaussian process added computational overhead to the MPC solve times, but it was still able to run onboard a Jetson TX2 computer.

[code](https://github.com/uzh-rpg/data_driven_mpc)

-----

*title:* Performance, Precision, and Payloads: Adaptive Nonlinear MPC for Quadrotors

Model-based classic control (3-C-1-124) / Hanover et al. [124] question whether or not it is necessary to explicitly model the additional aerodynamic terms from [40] due to the added computational and modeling complexity. Instead, they propose to learn residual model dynamics online using a cascaded adaptive NMPC architecture. Aggressive flight approaching 20m/s and over 4 g acceleration is demonstrated on real racing quadrotors. In addition, completely unknown payloads can be introduced to the system, with minimal degradation in tracking performance. The adaptive inner loop controller added minimal computational overhead and improved tracking performance over the Gaussian process MPC by 70% on a series of high-speed flights of a racing quadrotor [40], [124].

## IV - B: Deep GNC, Integrated (Learned) NG

## IV - C: Deep GNC, Learned C

## IV - D: Deep GNC, Integrated (Learned) GC

## IV - E: Deep GNC, Integrated (Learned) NGC

