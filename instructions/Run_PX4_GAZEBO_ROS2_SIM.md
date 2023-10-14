Multiple Vehicle with gazebo classic 
    • Tools/simulation/gazebo-classic/sitl_multiple_run.sh [-m <model>] [-n <number_of_vehicles>] [-w <world>] [-s <script>] [-t <target>] [-l <label>]


    • <model>: The vehicle type/model to spawn, e.g.: iris (default), plane, standard_vtol, rover, r1_rover typhoon_h480.
    • <number_of_vehicles>: The number of vehicles to spawn. Default is 3. Maximum is 254.
    • <world>: The world that the vehicle should be spawned into, e.g.: empty (default)

    • Supported vehicle types are: iris, plane, standard_vtol, rover, r1_rover typhoon_h480.
    • The number after the colon indicates the number of vehicles (of that type) to spawn.
    • Maximum number of vehicles is 254.
    • <target>: build target, e.g: px4_sitl_default (default), px4_sitl_nolockstep
    • <label> : specific label for model, e.g: rplidar

Example for multi vehicle 
```
PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_multiple_run.sh -m iris -n 2 -w empty 
```
