
# Object Detection and Tracking Literature Reviews

This is a set of brief reviews in the **Object Detection and Tracking** area on most recent research papers in the filed. 

Object tracking refers to the ability to estimate or predict the position of a target object in each consecutive frame in a video once the initial position of the target object is defined. On the other hand, object detection is the process of detecting a target object in an image or a single frame of the video.



## None-MDE-Aided Tracking (Control aspect)

Here, summarize the investigations in the new useful softwares which are included in an object detection and tracking project; Whether based on a motion model or visual data, or both.

We expect that most of the works listed in this part focus on the *guidance*, *navigation*, and *control* aspect of the problem in addition to the *computer vision* and *sensor fusion* algorithms.

After presenting the aforementioned data about each paper, answer to the following questions in your review:

1. Explain about the sensor fusion (if there is some)

2. Explain about the computer vision algorithm(s)

3. Explain about the navigation, guidance, and control strategy

4. Describe the performance in terms of technical parameters related to the tracking mission. 

5. Explain about the mission and the hardware


### Autonomous Human-Following Drone for Monitoring a Pedestrian from Constant Distance and Direction

[Link to the Paper](https://drive.google.com/file/d/1p6NsPXuzjZfd4ESUXRtnRvGcS1R0wn1F/view?usp=sharing)

[Link to the Source Code]()

**Authors:** 
Hiroto Yamashita, Takashi Morimoto, Ikuhisa Mitsugami

**Date:** 
2021

**Journal or Conference:** ...
2021 IEEE 10th Global Conference on Consumer Electronics (GCCE)

#### Review:

Distance evaluation: They evaluated the length between the Neck and Hip points in the captured image while walking and standing. the length was kept almost constant except for moments when the person started walking or stopped.
Direction evaluation: The drone moved to the position behind the subject again, although there was some delay in the movement change of the subject.

#### Answers:

1. Not used.

2. CNN-based 2D pose estimation by OpenPose and 3D pose estimation by “Martinez et al” method.

3. The 3D joint points are estimated from 2D points with different skeleton model.
PID controller is applied. input values are the longitudinal, movement speed, vertical movement speed, and rotational, angular speed.
The human direction, Theta is calculated. This system controls the left-right movement speed of the drone so that theta approaches the target value.

4.  It is a real-time system that estimate humans 2D and 3D pose and Control parameter are calculated from skeleton points.
Experimental results confirmed that our system ran with reasonable response time and positional stability.

5. An autonomous drone system that follows a pedestrian from a certain distance and direction for human video analysis without environmental constraints. +Tello drone +RGB camera


### Multiple human tracking in drone image

[Link to the Paper](https://link.springer.com/article/10.1007/s11042-018-6141-z)

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

1.

2.

3.

4.

5.


## MDE for Object Detection and Tracking

The applications of MDE in the object detection and tracking must be summarized here

Try to answer the folloeing questions in your review:

1. What are the drone type and config?

2. What's the mission? Tracking a car? Human? Describe how hard the mission is and the mission constraints. How, and with what details, is the system implemented?

3. What are the sensors or pseudo sensors? Is the system V-INS? Just Visual? Or something else?

4. What data are fused?

5. What is the overal method to detect or track an object?

6. If there is, discuss more about the object motion model 


Add the papers you find in the following:

### Depth Estimation Matters Most: Improving Per-Object Depth Estimation for Monocular 3D Detection and Tracking

[Link to the Paper](https://arxiv.org/pdf/2206.03666.pdf)

[Link to the Source Code]()

**Authors:** Longlong Jing, Ruichi Yu, Henrik Kretzschmar, Kang Li, Charles R. Qi, Hang Zhao, Alper Ayvaci, Xu Chen, Dillon Cower, Yingwei Li, Yurong You, Han Deng, Congcong Li, and Dragomir Anguelov, Waymo LLC, Johns Hopkins University, Cornell University

**Date:**  8 Jun 2022

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

1.

2.

3.

4.

5.

6.



### Deep Learning for Real-Time 3D Multi-Object Detection, Localisation, and Tracking: Application to Smart Mobility

[Link to the Paper](https://www.mdpi.com/1424-8220/20/2/532)

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

1.

2.

3.

4.

5.

6.



### Realtime Object-aware Monocular Depth Estimation in Onboard Systems


[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

1.

2.

3.

4.

5.

6.


### Monocular 3D Multi-Object Tracking with an EKF Approach for Long-Term Stable Tracks

[Link to the Paper](https://drive.google.com/file/d/1rJ3PnFiJbZcrKbh4VGLAxsz_62sTOKMX/view?usp=sharing)

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

1.

2.

3.

4.

5.

6.


### Real-time Monocular 3D People Localization and Tracking on Embedded System

[Link to the Paper](https://drive.google.com/file/d/1HKHRA1zM6AxbsaZgxfexzKUygNpzXZTo/view?usp=sharing)

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

1.

2.

3.

4.

5.

6.


### Design of a Robust System Architecture for Tracking Vehicle on Highway Based on Monocular Camera

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

1.

2.

3.

4.

5.

6.


### YOLO MDE: Object Detection with Monocular Depth Estimation

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

1.

2.

3.

4.

5.

6.
