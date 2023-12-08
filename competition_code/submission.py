"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )


    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
        # Dynamic waypoint viewer, depends on speed.
        lookahead_distance = np.floor(0.425 * vehicle_velocity_norm)
        waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + int(lookahead_distance)) % len(self.maneuverable_waypoints)]

        # Dynamic waypoint viewer (same as one above), BUT looks further ahead, providing more advanced warning (so the car doesn't slam into a wall)
        further_lookahead_distance = np.floor(0.90 * vehicle_velocity_norm)
        further_waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + int(further_lookahead_distance)) % len(self.maneuverable_waypoints)]

        # Dynamic waypoint viewer (same as one above), BUT looks EVEN further ahead, providing EVEN more advanced warning (so the car doesn't slam into a wall)
        superfar_lookahead_distance = np.floor(1.25 * vehicle_velocity_norm) #1.265
        superfar_waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + int(superfar_lookahead_distance)) % len(self.maneuverable_waypoints)]

        
        # waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + 3) % len(self.maneuverable_waypoints)]

        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta vector towards "advanced warning" target waypoint
        vector_to_further_waypoint = (further_waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_further_waypoint = np.arctan2(vector_to_further_waypoint[1],vector_to_further_waypoint[0])

        # Calculate delta vector towards "advanced warning" target waypoint
        vector_to_superfar_waypoint = (superfar_waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_superfar_waypoint = np.arctan2(vector_to_superfar_waypoint[1],vector_to_superfar_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Calculate the curavture of the upcoming track
        turning_radius = 1.0 / np.cross([vector_to_further_waypoint[0], vector_to_further_waypoint[1], 0], [np.cos(vehicle_rotation[2]), np.sin(vehicle_rotation[2]), 0])[2] if np.linalg.norm(vector_to_further_waypoint) > 0 else 0.0
        self.turning_radius = turning_radius # For testing
        curvature = 1.0 / turning_radius if turning_radius != 0 else 0.0
        self.curvature = curvature # For testing

        # Calculate the curavture of the track that is decently ahead of the vehicle
        superfar_turning_radius = 1.0 / np.cross([vector_to_superfar_waypoint[0], vector_to_superfar_waypoint[1], 0], [np.cos(vehicle_rotation[2]), np.sin(vehicle_rotation[2]), 0])[2] if np.linalg.norm(vector_to_further_waypoint) > 0 else 0.0
        superfar_curvature = 1.0 / superfar_turning_radius if superfar_turning_radius != 0 else 0.0

        # Proportional controller to steer the vehicle towards the target waypoint
        steer_control = (
            -18.0 / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        # Proportional controller to control the vehicle's speed towards 40 m/s

        # Define a proportional controller for speed adjustment based on curvature

        if (self.current_waypoint_idx % 2773 > 2600 and self.current_waypoint_idx % 2773 < 2725) or (self.current_waypoint_idx % 2773 > 1300 and self.current_waypoint_idx % 2773 < 1340): # Catching sharper curves that need more cautious control.
            if np.abs(superfar_curvature) > 20.0: # Big turn far ahead, slow down!
                throttle_control = -0.15
            else: # If the superfar curve detector doesn't detect a very big curve ahead...
                throttle_control = 1
        elif (self.current_waypoint_idx % 2773 > 390) or (self.current_waypoint_idx % 2784 < 450): # Catching sharp-ish curves that need more cautious control.
            if np.abs(superfar_curvature) > 20.0: # Big turn far ahead, slow down!
                throttle_control = 0.32
            else: # If the superfar curve detector doesn't detect a very big curve ahead...
                throttle_control = 1
        else:
            throttle_control = 1

        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(-throttle_control, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }

        await self.vehicle.apply_action(control)
        return control
