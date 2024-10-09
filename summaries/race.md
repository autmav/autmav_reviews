

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

*title-1:* OIL: Observational Imitation Learning
*title-2:* Learning a Controller Fusion Network by Online Trajectory Filtering for Vision-based UAV Racing

**Review**: (4-E-[182,183]) train a perception-planning network and a control network using imitation learning. The perception network takes raw images as input and predicts waypoints to the next gate. The control network uses such predictions with ground-truth velocity and attitude information to predict control commands for tracking the waypoints. They showed improvements over pure end-to-end approaches, which directly map pixels to control commands and were able to show competitive lap times on par with intermediate human pilots within the Sim4CV simulator [184].

[code-1](https://drive.google.com/file/d/10UoFPVcb7lENDQUokLCZEM4g8MAlqTmO/view)
