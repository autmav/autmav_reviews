Install PX4
You need to install the PX4 development toolchain in order to use the simulator.

```
cd 
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
cd PX4-Autopilot/
make px4_sitl

```
 Setup Micro XRCE-DDS Agent & Client
To setup and start the agent:
   1. Open a terminal.
   2. Enter the following commands to fetch and build the agent from source:
```
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
```

Start the agent
    3. Start the agent with settings for connecting to the uXRCE-DDS client running on the simulator:
```
     MicroXRCEAgent udp4 -p 8888
```


