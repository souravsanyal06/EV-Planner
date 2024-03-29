# Setup
EV_Planner is the catkin worspace directory. 

Rename it as cv_ws.

Major Implementations are available at src/trajectory_learner

Other ROS packages are drawn from the respective github sub-repos. 

git clone --recurse-submodules -j8 https://github.com/souravsanyal06/EV-Planner.git 

To build, run build.sh 

# Abstract
Vision-based object tracking is an essential precursor to performing autonomous aerial navigation in order to avoid obstacles. Biologically inspired neuromorphic event cameras are emerging as a powerful alternative to frame-based cameras, due to their ability to asynchronously detect varying intensities (even in poor lighting conditions), high dynamic range, and robustness to motion blur. Spiking neural networks (SNNs) have gained traction for processing events asynchronously in an energy-efficient manner. On the other hand, physics-based rtificial intelligence (AI) has gained prominence recently, as they enable embedding system knowledge via physical modeling inside traditional analog neural networks (ANNs). In this letter, we present an event-based physics-guided neuromorphic planner (EV-Planner) to perform obstacle avoidance using neuromorphic event cameras and physics-based AI. We consider the task of autonomous drone navigation where the mission is to detect moving gates and fly through them while avoiding a collision. We use event cameras to perform object detection using a shallow spiking neural network in an unsupervised fashion. Utilizing the physical equations of the brushless DC motors present in the drone rotors, we train a lightweight energy-aware physics- guided neural network (PgNN) with depth inputs. This predicts the optimal flight time responsible for generating near-minimum energy paths. We spawn the drone in the Gazebo simulator and implement a sensor-fused vision-to-planning neuro-symbolic framework using Robot Operating System (ROS). Simulation results for safe collision-free flight trajectories are presented with performance analysis, ablation study and potential future research directions.

# System Overview:
<img width="779" alt="image" src="https://github.com/souravsanyal06/EV-Planner/assets/33360350/f79f7ea3-2f49-4d6d-8390-6abae3382b7a">

# Publication Link

https://ieeexplore.ieee.org/abstract/document/10382663?casa_token=4mefcwX1EfMAAAAA:3Gg0PAjyNMS7njGjg4-UeEMJflSxXO2GSHlUCmdrwp3Fml5J9wa1efZKM9IrIlxUiXtAQFyC2fQ

# ArxiV Link:  
https://arxiv.org/pdf/2307.11349.pdf

# Slides with voice-over available at:
https://www.youtube.com/watch?v=NDOw2ItGGa0

# Event-based Tracking of moving Ring
![EV_box](https://github.com/souravsanyal06/EV-Planner/assets/33360350/ab0b8fe7-8ad0-4828-bf34-ec5372b8e842)



# Video demonstration of Obstacle Avoidance:
https://github.com/souravsanyal06/EV-Planner/assets/33360350/9ec065ac-8ebd-4bd5-8032-198fc16e363c


# Citation
S. Sanyal, R. K. Manna and K. Roy, "EV-Planner: Energy-Efficient Robot Navigation via Event-Based Physics-Guided Neuromorphic Planner," in IEEE Robotics and Automation Letters, doi: 10.1109/LRA.2024.3350982.


