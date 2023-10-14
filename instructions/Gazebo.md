Install gazebo 
```
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add –
sudo apt-get update
sudo apt-get upgrade
$ sudo apt-get update
...
Hit http://ppa.launchpad.net bionic/main Translation-en
Ign http://us.archive.ubuntu.com bionic/main Translation-en_US
Ign http://us.archive.ubuntu.com bionic/multiverse Translation-en_US
Ign http://us.archive.ubuntu.com bionic/restricted Translation-en_US
Ign http://us.archive.ubuntu.com bionic/universe Translation-en_US
Reading package lists... Done
sudo apt-get install gazebo11
```
 For developers that work on top of Gazebo, one extra package
```
sudo apt-get install libgazebo11-dev
```

####For adding your own world, you have to go to these locations and add your world:

./Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds (Add your world file here)
./src/modules/simulation/simulator_mavlink/sitl_targets_gazebo-classic.cmake (Add your world to Cmake list)

