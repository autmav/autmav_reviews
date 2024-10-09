

Organized review of the paper `Autonomous Drone Racing: A Survey, IEEE TRANSACTIONS ON ROBOTICS, VOL. 40, 2024`, with focus on modern guidance and control for agile quadrotors.


## III - B: Classic GNC, Guidance (Planning)

*db-num:* 0013

*title:* Fast-Racing: An Open-source Strong Baseline for SE(3) Planning in Autonomous Drone Racing

**Review:** (3-B-1-87) / Trajectory planning after finding a geometrical path between a given start and goal position and avoiding obstacle then uses a found geometric path to either create a collision-free flight corridor. 

**Abstract:** This paper proposes an open-source baseline, which includes a high-performance SE(3) planner and a challenging simulation platform tailored for drone racing. We specify the SE(3) trajectory generation as a soft-penalty optimization problem, and speed up the solving process utilizing its underlying parallel structure.

[code](https://github.com/ZJU-FAST-Lab/Fast-Racing)

-----

*db-num:* 0014

*title:* Minimum-time trajectory generation for quadrotors in constrained environments

**Review:** (3-B-1-88) / Trajectory planning after finding a geometrical path between a given start and goal position and avoiding obstacle then uses a found geometric path to either create a collision-free flight corridor.

**Abstract:** In this paper, a novel strategy to compute minimum-time trajectories for quadrotors in constrained environments. In particular, we consider the motion in a given flying region with obstacles and take into account the physical limitations of the vehicle.

-----

*db-num:* 0015

*title:* Minimum-Time Quadrotor Waypoint Flight in Cluttered Environments
**Review:** (3-B-1-37) / Find new waypoints for the trajectory to avoid collisions.

**Abstract:** In the Article they tackle the problem of planning a minimum-time trajectory for a quadrotor over a sequence of specified waypoints in the presence of obstacles while exploiting the full quadrotor dynamics.

[code](https://github.com/uzh-rpg/sb_min_time_quadrotor_planning)

-----

*db-num:* 0016

*title:* Polynomial Trajectory Planning for Aggressive Quadrotor Flight in Dense Indoor Environments
**Review:** (3-B-1-89) / Find new waypoints for the trajectory to avoid collisions.

**Abstract:** They explore the challenges of planning trajectories for quadrotors through cluttered indoor environments. They extend the existing work on polynomial trajectory generation by presenting a method of jointly optimizing polynomial path segments in an unconstrained quadratic program that is numerically stable for high order polynomials and large numbers of segments, and is easily formulated for efficient sparse computation.

[code](https://github.com/ethz-asl/mav_trajectory_generation)

-----

*db-num:* 0017

*title:* Robust Real-time UAV Replanning Using Guided Gradient-based Optimization and Topological Paths
**Review:** (3-B-1-90) / Constrain the trajectory to stay close to the found path.

**Abstract-1:** Gradient-based trajectory optimization suffers from local minima, which is not only fatal to safety but also unfavorable for smooth navigation. In this paper, They propose a replanning method based on GTO addressing this issue systematically. A path-guided optimization (PGO) approach is devised to tackle infeasible local minima, which improves the replanning success rate significantly.

*db-num:* 0018

*title:* RAPTOR: Robust and perception aware trajectory replanning for quadrotor fast flight
**Review:** (3-B-1-91) / Constrain the trajectory to stay close to the found path.

**Abstract-2:** Recent advances in trajectory replanning have enabled quadrotor to navigate autonomously in unknown environments. However,high-speed navigation still remains a significant challenge. Given very limited time, existing methods have no strong guarantee on the feasibility or quality of the solutions. Moreover, most methods do not consider environment perception, which is the key bottleneck to fast flight. In this paper, They present RAPTOR, a robust and perception-aware replanning framework to support fast and safe flight, which addresses these issues systematical

[code-1](https://github.com/HKUST-Aerial-Robotics/TopoTraj)
[code-1,2](https://github.com/HKUST-Aerial-Robotics/Fast-Planner)

-----

*db-num:* 0019

*title:* Vision-Based Reactive Planning for Aggressive Target Tracking while Avoiding Collisions and Occlusions
**Review:** (3-B-1-94) / Some works rely solely on trajectory planning as they assume no collision with the environment when a trajectory is found.

**Abstract:** In this paper they investigate the online generation of optimal trajectories for target tracking with a quadrotor while satisfying a set of image-based and actuation constraints. We consider a quadrotor equipped with a camera (either down or front-looking) with limited field of view.

-----

*db-num:* 0020

*title:* Time-optimal planning for quadrotor waypoint flight
**Review:** (3-B-1-95) / Some works rely solely on trajectory planning as they assume no collision with the environment when a trajectory is found.

**Abstract:** The paper proposes a method to generate time-optimal trajetories for quadrotors by solving the time-allocation problem while maximizing the use of the quadrotore's actuators. This approach outperforms existing methods and even human pilos in drone-racing tasks.

[code](https://github.com/uzh-rpg/rpg_time_optimal)

-----

*db-num:* 0021

*title:* Fast Trajectory Optimization for Agile Quadrotor Maneuvers with a Cable-Suspended Payload
**Review:** (3-B-1-96) / Some works rely solely on trajectory planning as they assume no collision with the environment when a trajectory is found.

**Abstract:** They present a novel dynamical model and a fast trajectory optimization algorithm for quadrotors with a cable-suspended payload. Theire first contribution is a new formulation of the suspended payload behavior, modeled as a link attached to the quadrotor with a combination of two revolute joints and a prismatic joint, all being passive.

-----

*db-num:* 0022

*title:* Performance benchmarking of quadrotor systems using time-optimal control
**Review:** (3-B-1-97) / Some works rely solely on trajectory planning as they assume no collision with the environment when a trajectory is found.

**Abstract:** This paper presents an algorithm that allows the computation of quadrotor maneuvers that satisfy Pontryagin’s minimum principle with respect to time-optimality.Such maneuvers provide a useful lower bound on the duration of maneuvers, which can be used to assess performance of controllers and vehicle design parameters.

-----

*db-num:* 0023

*title:* 4D Trajectory Prediction with Model Predictive Control based on Flight Plan
**Review:** (3-B-1-98) / Some works rely solely on trajectory planning as they assume no collision with the environment when a trajectory is found.

**Abstract:** The paper aims to improve air traffic management by predicting the four-dimensional trajectory (4DT) of aircraft, which includes their three-dimensional position over time. This prediction is crucial for ensuring safe and efficient operations, especially in urban air mobility (UAM) environments. The study proposes a method using flight plans and Model Predictive Control (MPC) to generate 4DTs, considering constraints like speed and altitude. The generated 4DTs were validated against actual flight data, showing potential to enhance air traffic efficiency and safety. 

-----

*db-num:* 0024

*title:* Search-based motion planning for quadrotors using linear quadratic minimum time control
**Review:** (3-B-1-99) / Works directly find a collision-free trajectory.

**Abstract:** In this work, they propose a search-based planning method to compute dynamically feasible trajectories for a quadrotor flying in an obstacle-cluttered environment. Theire approach searches for smooth, minimum-time trajectories by exploring the map using a set of short-duration motion primitives.

[code](https://github.com/sikang/mpl_ros)

-----

*db-num:* 0025

*title:* Search-based motion planning for aggressive flight in SE(3)
**Review:** (3-B-1-100) / Works directly find a collision-free trajectory.

**Abstract:** The article aims to develop a trajectory planning method for quadrotors with high thrust-to-weight ratios, enabling them to navigate cluttered environments with sharp turns and high accelerations. Using an ellipsoid model to check for collisions, the approach finds safe, optimal paths through narrow gaps by adjusting pitch and roll. It accelerates planning with a lower-dimensional search and demonstrates effectiveness in simulations and real-world tests.

[code](https://github.com/StanfordASL/KinoFMT.git)

-----

*db-num:* 0026

*title:* A real-time framework for kinodynamic planning with application to quadrotor obstacle avoidance
**Review:** (3-B-1-101) / Works directly find a collision-free trajectory.

**Abstract:** The objective of this paper is to present a full-stack, real-time kinodynamic planning framework and demonstrate it on a quadrotor for collision avoidance. Specifically, the proposed framework utilizes an offline online computation paradigm, neighborhood classification through machine learning, sampling-based motion planning with an optimal control distance metric, and trajectory smoothing to achieve real-time planning for aerial vehicles.

[code](https://github.com/StanfordASL/KinoFMT.git)

-----

*db-num:* 0027

*title:* Vector field guided RRT* based on motion primitives for quadrotor kinodynamic planning
**Review:** (3-B-1-102) / Works directly find a collision-free trajectory.

**Abstract:**  In this paper, they present a sampling-based kinodynamic planning algorithm for
quadrotors, which plans a dynamically feasible trajectory in a complicated environment. We have designed a method to constrain the sampling state by using the vector field to construct a cone in the sampling stage of RRT*, so that the generated trajectory is connected as smoothly as possible to other states in the reachable set.

[pesaudocode] (4.1 in paper)

-----

*db-num:* 0028

*title:* Model predictive contouring control for time-optimal quadrotor flight
**Review:** (3-B-1-103) / Some control approaches do not need a specified time-allocated trajectory and rely only on the geometrical path for controlling the drone.

**Abstract:** This article presents a real-time solution for flying time-optimal trajectories with quadrotors using Model Predictive Contouring Control (MPCC). It efficiently combines trajectory planning and control, outperforming traditional methods and even a world-class human pilot in real-world tests.

-----

*db-num:* 0029

*title:* Time-optimal online replanning for agile quadrotor flight
**Review:** (3-B-1-104) / Some control approaches do not need a specified time-allocated trajectory and rely only on the geometrical path for controlling the drone.

**Abstract:** The article presents a method for real-time, time-optimal control of quadrotors that can adapt to changes in the environment or disturbances. Using a sampling-based approach combined with Model Predictive Contouring Control (MPCC), the system allows the quadrotor to replan its path on-the-fly, demonstrated in high-speed flights and strong winds.

-----

*db-num:* 0030

*title:* Minimum snap trajectory generation and control for quadrotors
**Review:** (3-B-1-111) / 

**Abstract:** The article focuses on designing a controller and generating optimal trajectories for a quadrotor navigating tight indoor environments. It introduces an algorithm that enables real-time generation of safe, optimal trajectories while accounting for constraints like velocities and accelerations. A nonlinear controller is used for accurate trajectory tracking, with experimental results demonstrating its effectiveness in fast 3D movements through slalom courses.

[code](https://github.com/ardyadipta/quadrotor_autonomy/tree/master/papers)

-----

*db-num:* 0031

*title:* A computationally efficient motion primitive for quadrocopter trajectory generation
**Review:** (3-B-1-113) / The polynomial and spline methods leverage the differential flatness property of quadrotors and represent a trajectory as a continuous-time polynomial or spline

**Abstract:** The article presents a fast method to generate and verify motion primitives for quadcopters, optimizing movements for tasks like catching a ball. It enables quick evaluations of thousands of motion options per second using computationally efficient algorithms.

-----

*db-num:* 0032

*title:* “Time-optimal gate traversing planner for autonomous drone racing
**Review:** (3-B-1-114) / Recently they in this paper they proposed a polynomial trajectory representation based on [115] and use it to plan time-optimal trajectories through gates of arbitrary shapes for drone racing, achieving close-to-time-optimal results while being more computationally efficient than [95].

**Abstract:** The purpose of this article is to introduce a time-optimal trajectory planner for drone racing that accounts for the configuration of race gates, unlike previous studies that treat gates as simple waypoints. The planner generates faster, more efficient trajectories by fully utilizing gate shapes and sizes, while considering the drone's thrust limits. It is computationally efficient and has been validated in simulations and real-world experiments, reducing lap times and enabling extreme flight trajectories.
[code](https://github.com/FSC-Lab/TOGT-Planner)

-----

*db-num:* 0033

*title:* Geometrically Constrained Trajectory Optimization for Multicopters
**Review:** (3-B-1-115) / A polynomial trajectory representation

**Abstract:** The purpose of this article is to introduce an optimization-based framework for multicopter trajectory planning that handles geometrical and dynamic constraints efficiently. The framework uses a novel trajectory representation to minimize control effort, and employs smooth maps to handle constraints in a lightweight manner. By transforming constrained problems into unconstrained optimization tasks, it ensures high-quality solutions while maintaining computational speed. The approach is demonstrated through various flight tasks, simulations, and benchmarks, showcasing its generality, robustness, and efficiency compared to other methods.

[code](https://github.com/ZJU-FAST-Lab/GCOPTER)

-----

*db-num:* 0034

*title:* Euclidean and non-Euclidean trajectory optimization approaches for quadrotor racing
**Review:** (3-B-1-116) / Direct collocation methods that rely on polynomials to approximate the input and state dynamics can achieve nearly optimal performance.

**Abstract:** The purpose of this article is to introduce two optimization approaches for quadrotor raceline planning that utilize either Euclidean or non-Euclidean geometry to describe vehicle position. Both methods leverage high-fidelity quadrotor dynamics and eliminate the need for approximating gates with waypoints. The article demonstrates significantly faster computation times 100 times faster than comparable methods while also enhancing solver convergence. Additionally, it extends the non-Euclidean approach to calculate racelines in environments with multiple static obstacles.

[code](https://zenodo.org/record/5036287)

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
*db-num:* 0001
*title:* **Performance, Precision, and Payloads: Adaptive Nonlinear MPC for Quadrotors**

Model-based classic control (3-C-1-124) / Hanover et al. [124] question whether or not it is necessary to explicitly model the additional aerodynamic terms from [40] due to the added computational and modeling complexity. Instead, they propose to learn residual model dynamics online using a cascaded adaptive NMPC architecture. Aggressive flight approaching 20m/s and over 4 g acceleration is demonstrated on real racing quadrotors. In addition, completely unknown payloads can be introduced to the system, with minimal degradation in tracking performance. The adaptive inner loop controller added minimal computational overhead and improved tracking performance over the Gaussian process MPC by 70% on a series of high-speed flights of a racing quadrotor [40], [124].

-----

*db-num:* 0002

*title-1:* **Model Predictive Contouring Control for Time-Optimal Quadrotor Flight**
*title-main:* **Time-Optimal Online Replanning for Agile Quadrotor Flight**

Model-based classic control (3-C-1-103 and 104) / model predictive contouring control
(MPCC) MPCC was then extended to agile quadrotor flight in [103]. Although the velocities achieved by the MPCC controller were lower than that of [124] and [125], the lap times for the same race track were actually lower due to the ability of the controller to find a new time allocation that takes into account the current state of the platform at every timestep. The work is further extended to solve the time-allocation problem online, and to replan online [104] while also controlling near the limit of the flight system.

-----

*db-num:* 0003

*title:* **Towards Time-Optimal Tunnel-Following for Quadrotors**

Model-based classic control (3-C-1-134) / Similar work uses tunneling constraints in the MPCC formulation in [134].

-----

## IV - B: Deep GNC, Integrated (Learned) NG

## IV - C: Deep GNC, Learned C

*db-num:* 0004

*title:* **Reinforcement Learning for UAV Attitude Control**

Model-based classic control (4-C-168) / model-free RL was applied to low-level attitude control [168], in which a learned low-level controller trained with proximal policy optimization (PPO) outperformed a fully tuned PID controller on almost every metric.

[code-1](https://github.com/wil3/gymfc)
[code-2](https://github.com/wil3/neuroflight)

-----

*db-num:* 0005

*title:* Low-Level Control of a Quadrotor With Deep Model-Based Reinforcement Learning

Model-based classic control (4-C-169) / Lambert et al. [169] used model-based RL for low-level control of an a priori unknown dynamic system

[code-1](https://github.com/natolambert/ros-crazyflie-mbrl)
[code-2](https://github.com/natolambert/crazyflie-firmware-pwm-control)
[code-3](https://github.com/natolambert/dynamicslearn)

-----

*db-num:* 0006

*title:* **A Benchmark Comparison of Learned Control Policies for Agile Quadrotor Flight**

*year:* 2022

**Review**: Model-based classic control (4-C-36) / recent works showcased the potential of learning-based controllers for high-speed trajectory tracking and drone racing

**Abstract**: (i) we show that training a control policy that commands body-rates and thrust results in more robust sim-to-real transfer compared to a policy that directly specifies individual rotor thrusts, (ii) we demonstrate that such a control policy trained via drl  can control a quadrotor in real-world experiments at speeds over 45 km/h.

-----

*db-num:* 0007

*title:* **Aggressive Online Control of a Quadrotor via Deep Network Representations of Optimality Principles**

*year:* 2020

**Review**: Model-based classic control (4-C-170) / Imitation learning is more data efficient compared to model-free RL. In [170], aggressive online control of a quadrotor has been achieved via training a network policy offline to imitate the control command produced by a model-based controller

**Abstract**: ...

-----

*db-num:* 0008

*title:* **End-to-end neural network based optimal quadcopter control**

*year:* 2023

**Review**: Model-based classic control (4-C-172) / has shown that RL can find optimal controllers

**Abstract**: ...

[code](https://github.com/tudelft/optimal_quad_control_SL)

-----

*db-num:* 0009

*title:* **Deep Model Predictive Optimization**

*year:* 2023 (arxiv) - 2024 (conf.)

**Review**: Model-based classic control (4-C-173) / has shown that RL can find optimal [122], [172] or highly adaptive controllers

**Abstract**: ...

[results](https://tinyurl.com/mr2ywmnw)

-----

*db-num:* 0010

*title:* **Lyapunov-stable neural-network control**

*year:* 2021

**Review**: Model-based classic control (4-C-174) / With a learning-based controller, it can be difficult to provide robustness guarantees as with traditional methods such as the linear quadratic regulator (LQR). While a learning-based controller may provide superior performance to classical methods in simulation, it may be that they cannot be used in the real world due to the inability to provide an analysis of the controller’s stability properties. This is particularly problematic for tracking the time-optimal trajectories required by drone racing. Recent works have attempted to address this using the Lyapunov-stable neural network design for the control of quadrotors [174]. This work shows that it is possible to have a learning-based controller with guarantees that can also outperform classical LQR methods.

**Abstract**: ...

[code](https://github.com/StanfordASL/neural-network-lyapunov)

-----

*db-num:* 0011

*title:* **Safe Reinforcement Learning Using Black-Box Reachability Analysis**

*year:* 2022

**Review**: Model-based classic control (4-C-175) / Building upon this concept (explained about 174), reachability analysis and safety checks can be embedded in a learned safety layer

**Abstract**: ...

[code](https://github.com/Mahmoud-Selim/Safe-Reinforcement-Learning-for-Black-Box-Systems-Using-Reachability-Analysis)

-----

*db-num:* 0012

*title:* **Autonomous Drone Racing with Deep Reinforcement Learning**

*year:* 2021

**Review**: Model-based classic control (4-C-122) / has shown that RL can find optimal controllers. where a neural network policy is trained with RL to fly through a race track in simulation in near-minimum time.

**Abstract**: ...



## IV - D: Deep GNC, Integrated (Learned) GC

-----

*db-num:* 

*title:* Champion-level drone racing using deep reinforcement learning

*year:* 2023

**Review**: (4-D-[5]) Producing the control command directly from state inputs without requiring a high-level trajectory planner, enabled an autonomous drone with only onboard perception, for the first time, to outperform a professional human, and is state-of-the-art at the time of writing. 

**Abstract**: ...

[pseudo-code](https://zenodo.org/records/7955278)

-----

*db-num:*

*title:* Reaching the Limit in Autonomous Racing: Optimal Control versus Reinforcement Learning

*year:* 2023

**Review**: (4-D-[4]) / A neural network policy is trained with RL to fly through a race track in simulation in near-minimum time.

**Abstract**: ...

-----

*db-num:* 

*title:*  Learning minimum-time flight in cluttered environments

*year:* 2022

**Review**: In [123], deep RL is combined with classical topological path planning to train robust neural network controllers for minimum-time quadrotor flight in cluttered environments. The learned policy solves the planning and control problem simultaneously, forgoing the need for explicit trajectory planning and control.

**Abstract**: ...

-----

*db-num:* 

**Review**: (4-D-[176,177,178,179,180]) / Another class of algorithms try to exploit the benefits of model-based and learning-based approaches using differentiable optimizers approaches [176], [177], [178], which leverage differentiability through controllers. For example, for tuning linear controllers by getting the analytic gradients [179], or for creating a differentiable prediction, planning,
and controller pipeline for autonomous vehicles [180].

*title-1:* Differentiable MPC for End-to-end Planning and Control
*year-1:* 2018

[code-1](https://github.com/locuslab/mpc.pytorch)
[code-1](https://github.com/locuslab/differentiable-mpc)

*title-2:* Theseus: A Library for Differentiable Nonlinear Optimization
*year-2:* 2022

[code-2](https://github.com/facebookresearch/theseus)

*title-3:* PyPose: A Library for Robot Learning with Physics-based Optimization
*year-3:*  2023

*title-4:* DiffTune+: Hyperparameter-Free Auto-Tuning using Auto-Differentiation
*year-4:* 2023

[code-4](https://github.com/Sheng-Cheng/DiffTuneOpenSource)

*title-5:* DiffStack: A Differentiable and Modular Control Stack for Autonomous Vehicles
*year-5:*  2023

[code-5](https://github.com/NVlabs/diffstack)

-----

*db-num:* 

*title:* Actor-Critic Model Predictive Control

*year:* 2024

**Review**: Romero et al. [181] equip the RL agent with a differentiable MPC [176], located at the last layer of the actor network that provides the system with online replanning capabilities and allows the policy to predict and optimize the short-term consequences of its actions while retaining the benefits of RL training.

**Abstract**: ...


-----

## IV - E: Deep GNC, Integrated (Learned) NGC

*db-num:* 0035

*title-1:* OIL: Observational Imitation Learning
*title-2:* Learning a Controller Fusion Network by Online Trajectory Filtering for Vision-based UAV Racing

**Review**: (4-E-[182,183]) train a perception-planning network and a control network using imitation learning. The perception network takes raw images as input and predicts waypoints to the next gate. The control network uses such predictions with ground-truth velocity and attitude information to predict control commands for tracking the waypoints. They showed improvements over pure end-to-end approaches, which directly map pixels to control commands and were able to show competitive lap times on par with intermediate human pilots within the Sim4CV simulator [184].

[code-1](https://drive.google.com/file/d/10UoFPVcb7lENDQUokLCZEM4g8MAlqTmO/view)

-----

*db-num:* 0035

*title:* Teaching UAVs to race: End-to-end regression of agile controls in simulation

**Review**: (4-E-[185]) The second family of approaches directly maps sensor observation to commands without any modularity. This design is used in [185], which to date remains the only example of the completely end-to-end racing system.

-----

*db-num:* 0036

*title:* DeepPilot: A CNN for Autonomous Drone Racing

**Review**: (4-E-[186]) other end-to-end systems generally require an inner loop controller and inertial information to be executed. For instance, Rojas-Perez and Martinez-Carranza [186] train an end-to-end CNN to directly predict roll, pitch, yaw, and altitude from camera images.

[code](https://github.com/QuetzalCpp/DeepPilot)

-----
*db-num:* 0037

*title-1:* Learning deep sensorimotor policies for vision-based autonomous drone racing
*title-2:* Contrastive learning for enhancing robust scene transfer in vision-based agile flight

**Review**: authors in [187] and [188] use a neural network to predict commands directly from vision. To improve sample complexity, they use contrastive learning to extract robust feature representations from images and leverage a two-stage learning-by-cheating framework.


