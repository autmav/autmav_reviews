
# Drone Localization Literature Reviews

The purpose is to review the papers focusing on localization of drones in a limited indoor environment. The works to be reviewd must use camera sensors as exterior sensors. They may also use the interior IMU sensors mounted on the flying robots to fuse them with the camera data in the localization algorithm.

Answer to the following questions in your reviews:

1. How many cameras are used externally?

2. Is IMU sensor used?

3. How is the object detected? Explain the whole computer vision scheme.

3. Explain the posotioning algorithm *exactly*

4. What sensor fusion paradigms are followed (if it's been)?

5. Is the object's motion model encountered in the algorithm? How?

6. How many drones are localized? Is the algorithm able to localize more than one object?

7. What is the speed limit for the moving robots in the implementation?

8. Explain the implementation platform.



### Detecting, Tracking, and Localizing a Moving Quadcopter Using Two External Cameras

[Link to the Paper](https://www.researchgate.net/publication/325963801_Detecting_Tracking_and_Localizing_a_Moving_Quadcopter_Using_Two_External_Cameras)

*NO SOURCE CODE*

**Authors:** Matthew Dreyer, Sugandh Raj, Srikanth Gururajan, and Jeremy Glowacki

**Date:** 2018

**Conference:** ...

#### Review:


#### Answers:

1.

2.

3.

...




#### General Moving Object Localization From a Single Flying camera 

[Link to the Paper](https://www.mdpi.com/2076-3417/10/19/6945/pdf?version=1601801770)

**Authors** Kin-Choong Yow , Insu Kim

**Date** 2020

**Conference** Journal 




#### Answers 

1. How many cameras are used externally? single moving camera 

2. Is IMU sensor used? no

3. How is the object detected? Explain the whole computer vision scheme. 

3. Explain the posotioning algorithm *exactly*

4. What sensor fusion paradigms are followed (if it's been)?

5. Is the object's motion model encountered in the algorithm? How?

6. How many drones are localized? Is the algorithm able to localize more than one object?

7. What is the speed limit for the moving robots in the implementation?

8. Explain the implementation platform.

#### Unmanned aerial vehicle localization using distributed sensors 

[Link to the Paper](https://www.researchgate.net/publication/320018288_Unmanned_aerial_vehicle_localization_using_distributed_sensors)

**Authors:** Seung Yeob Nam,Gyanendra Prasad Joshi

**Date:** 30 August 2017

**Conference:** Journal


1. How many cameras are used externally? 

In this approach, several cameras connected through sensor networks usually play the role of the vision sensor to detect UAV by image processing.

2. Is IMU sensor used? 

IMU sensor is not used in this approach.

3. How is the object detected? Explain the whole computer vision scheme. 

In this approach, each sensor measures two angles, the azimuth angle and the elevation angle of the target. This measured information will then be sent to the centralized collector (sink) node through the sensor network. In the next step, the sink node estimates the location of the UAV based on the collected angle value. To simplify the problem, the following assumptions were made.

    • each sensor node has the ability to detect an aerial moving object and extract the two mentioned angles using image processing techniques. 

    • The clocks of the sensor nodes are synchronized. In result, all of the sensor nodes can take a picture at the same time. 

    • The entire set of sensor nodes shares a common coordinate system. They are positioned in the same direction and know the position of each sensor.

4. Explain the posotioning algorithm *exactly*.

In the first method, the location of the UAV is found by the following steps:

    1. The location of the UAV on the horizontal two-dimensional space using azimuth angles is estimated. 
    2. The altitude of the UAV using the measured angles of elevation is estimated. 
    3. The equation of the line emanating from the sensor toward the UAV can be obtained by the azimuth angle.
    4. The equation of the line passing the other sensor is obtained in a similar manner, thus the coordination of the target can be obtained from the intersection of these two lines. If the number of sensor nodes is three or more, the azimuth angle measurement is not accurate due to noise. It is not likely to be a single point where all the lines from the sensor nodes meet. In such a case, we estimate the location of the UAV by minimizing the summation of square of the distance between the selected point and the lines from each sensor node. 

In the second method, the estimation is done directly in the 3D space. Followed by the location of the UAV which is estimated by lines that pass the sensor, the azimuth angle and the elevation angle measured by sensors. The distance between an arbitrary point and the mentioned line is measured with the same calculation method mentioned above in a 3D space. 

5. What sensor fusion paradigms are followed (if it's been)?

In this approach, the only external sensors are cameras.

6. How many drones are localized? Is the algorithm able to localize more than one object?

One UAV is localized in this article.

This is not mentioned in this article.

7. What is the speed limit for the moving robots in the implementation?

This is not mentioned in this article.

8. Explain the implementation platform.

The proposed schemes are evaluated in the simulation environments.

 Two types of UAV are considered:

    • a high-altitude UAV, where the altitude is 550 m 
    • a low-altitude UAV, where the altitude is 50 m. 
    
 The number of sensors, M, changes from 4 to 30, and when the number of sensors is selected, the       position of each sensor is uniformly selected from the 3D range (0, 1000) 3 (0, 100) 3 (0, 10). However, the position of the target UAV is fixed at (500, 200, 50) for the low-altitude UAV and fixed at (500, 200, 550) for the high-altitude UAV. The relative position between the target UAV and each sensor is randomized by random selection of sensor positions. Also, the measurement error for azimuth angle and elevation angle is considered. Both of them are modelled as a Gaussian random variable with the same zero mean and a standard deviation of degrees. As a result, the accuracies of the 2D, 3D situation and centroid schemes for a low-altitude target UAV and various numbers of sensors are compared.