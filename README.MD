# Omicopter running
git clone https://github.com/PX4/PX4-Autopilot.git --recursive   

make px4_sitl gz_omnicopter

param set COM_OBL_RC_ACT 0 
param set COM_RCL_EXCEPT 4 
param set NAV_DLL_ACT 0
param save


-----CHECK Status--------
commander check
listener vehicle_status

listener actuator_outputs
listener vehicle_control_mode
listener actuator_armed


ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888

------ ENABLE CONTROL ALLOCATION -------------
modify this code: 
```PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/4001_gz_x500```

```bash
# control allocation pipeline for a quadrotor.
param set-default CA_AIRFRAME 1 
```
check by this command:
```
param show CA_AIRFRAME --> Should see: CA_AIRFRAME [ ]: 1
```