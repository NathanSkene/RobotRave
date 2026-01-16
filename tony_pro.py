#!/usr/bin/env python3
"""
Tony Pro Servo Configuration and Control Module.

This module provides the correct servo mapping for the Tony Pro robot
and utility functions for all dance scripts to use.

Usage:
    from tony_pro import SERVO, MOVES, get_neutral, TonyProController
"""

import json
import os
import sys

# =============================================================================
# SERVO ID MAPPING - Tony Pro 16-DOF Humanoid
# =============================================================================

# Servo ID constants for easy reference
class SERVO:
    """Tony Pro servo IDs by joint name."""

    # Head (2 DOF)
    HEAD_PITCH = 8      # Up/down
    HEAD_YAW = 16       # Left/right

    # Right Arm (2 DOF)
    R_SHOULDER = 7      # Shoulder pitch
    R_ELBOW = 5         # Elbow pitch

    # Left Arm (2 DOF)
    L_SHOULDER = 15     # Shoulder pitch
    L_ELBOW = 13        # Elbow pitch

    # Right Leg (5 DOF)
    R_HIP_ROLL = 1
    R_HIP_YAW = 2
    R_HIP_PITCH = 4
    R_KNEE = 3
    R_ANKLE = 6

    # Left Leg (5 DOF)
    L_HIP_ROLL = 9
    L_HIP_YAW = 10
    L_HIP_PITCH = 12
    L_KNEE = 11
    L_ANKLE = 14

# Grouped by body part for convenience
HEAD_SERVOS = [SERVO.HEAD_PITCH, SERVO.HEAD_YAW]
RIGHT_ARM_SERVOS = [SERVO.R_SHOULDER, SERVO.R_ELBOW]
LEFT_ARM_SERVOS = [SERVO.L_SHOULDER, SERVO.L_ELBOW]
RIGHT_LEG_SERVOS = [SERVO.R_HIP_ROLL, SERVO.R_HIP_YAW, SERVO.R_HIP_PITCH, SERVO.R_KNEE, SERVO.R_ANKLE]
LEFT_LEG_SERVOS = [SERVO.L_HIP_ROLL, SERVO.L_HIP_YAW, SERVO.L_HIP_PITCH, SERVO.L_KNEE, SERVO.L_ANKLE]

ARM_SERVOS = RIGHT_ARM_SERVOS + LEFT_ARM_SERVOS
LEG_SERVOS = RIGHT_LEG_SERVOS + LEFT_LEG_SERVOS
UPPER_BODY_SERVOS = ARM_SERVOS + HEAD_SERVOS
ALL_SERVOS = list(range(1, 17))

# ID to name mapping
SERVO_NAMES = {
    1: "right_hip_roll",
    2: "right_hip_yaw",
    3: "right_knee_pitch",
    4: "right_hip_pitch",
    5: "right_elbow_pitch",
    6: "right_ankle_pitch",
    7: "right_shoulder_pitch",
    8: "head_pitch",
    9: "left_hip_roll",
    10: "left_hip_yaw",
    11: "left_knee_pitch",
    12: "left_hip_pitch",
    13: "left_elbow_pitch",
    14: "left_ankle_pitch",
    15: "left_shoulder_pitch",
    16: "head_yaw",
}

# Name to ID mapping
SERVO_IDS = {v: k for k, v in SERVO_NAMES.items()}

# =============================================================================
# PULSE VALUES AND RANGES
# =============================================================================

PULSE_CENTER = 500
PULSE_MIN = 0
PULSE_MAX = 1000

# Safe operating ranges per servo (can be tuned after testing)
# Format: {servo_id: (min_pulse, max_pulse)}
SERVO_LIMITS = {
    # Head - moderate range for safety
    SERVO.HEAD_PITCH: (350, 650),
    SERVO.HEAD_YAW: (300, 700),

    # Arms - wider range for dancing
    SERVO.R_SHOULDER: (200, 800),
    SERVO.R_ELBOW: (200, 800),
    SERVO.L_SHOULDER: (200, 800),
    SERVO.L_ELBOW: (200, 800),

    # Legs - conservative range to prevent falls
    SERVO.R_HIP_ROLL: (400, 600),
    SERVO.R_HIP_YAW: (400, 600),
    SERVO.R_HIP_PITCH: (400, 600),
    SERVO.R_KNEE: (400, 600),
    SERVO.R_ANKLE: (400, 600),
    SERVO.L_HIP_ROLL: (400, 600),
    SERVO.L_HIP_YAW: (400, 600),
    SERVO.L_HIP_PITCH: (400, 600),
    SERVO.L_KNEE: (400, 600),
    SERVO.L_ANKLE: (400, 600),
}


def clamp_pulse(servo_id, pulse):
    """Clamp pulse value to safe range for given servo."""
    if servo_id in SERVO_LIMITS:
        min_p, max_p = SERVO_LIMITS[servo_id]
        return max(min_p, min(max_p, pulse))
    return max(PULSE_MIN, min(PULSE_MAX, pulse))


def get_neutral():
    """Get neutral pose (all servos at center)."""
    return {servo_id: PULSE_CENTER for servo_id in ALL_SERVOS}


# =============================================================================
# PRE-DEFINED DANCE MOVES
# =============================================================================

MOVES = {
    'neutral': {
        SERVO.HEAD_PITCH: 500, SERVO.HEAD_YAW: 500,
        SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
        SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
    },

    # Arm moves
    'arms_up': {
        SERVO.R_SHOULDER: 300, SERVO.R_ELBOW: 500,
        SERVO.L_SHOULDER: 700, SERVO.L_ELBOW: 500,
        SERVO.HEAD_PITCH: 400,
    },
    'arms_out': {
        SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 300,
        SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 700,
    },
    'arms_down': {
        SERVO.R_SHOULDER: 700, SERVO.R_ELBOW: 500,
        SERVO.L_SHOULDER: 300, SERVO.L_ELBOW: 500,
    },

    # Punch moves
    'right_punch': {
        SERVO.R_SHOULDER: 350, SERVO.R_ELBOW: 300,
        SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
        SERVO.HEAD_YAW: 600,
    },
    'left_punch': {
        SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
        SERVO.L_SHOULDER: 650, SERVO.L_ELBOW: 700,
        SERVO.HEAD_YAW: 400,
    },

    # Wave moves
    'wave_right': {
        SERVO.R_SHOULDER: 300, SERVO.R_ELBOW: 400,
        SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
        SERVO.HEAD_YAW: 550, SERVO.HEAD_PITCH: 450,
    },
    'wave_left': {
        SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
        SERVO.L_SHOULDER: 700, SERVO.L_ELBOW: 600,
        SERVO.HEAD_YAW: 450, SERVO.HEAD_PITCH: 450,
    },

    # Head moves
    'head_bob_down': {
        SERVO.HEAD_PITCH: 550,
        SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
        SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
    },
    'head_bob_up': {
        SERVO.HEAD_PITCH: 400,
        SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
        SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
    },
    'head_left': {
        SERVO.HEAD_YAW: 400,
    },
    'head_right': {
        SERVO.HEAD_YAW: 600,
    },

    # Celebration
    'celebrate': {
        SERVO.R_SHOULDER: 250, SERVO.R_ELBOW: 400,
        SERVO.L_SHOULDER: 750, SERVO.L_ELBOW: 600,
        SERVO.HEAD_PITCH: 350,
    },

    # Sway moves (subtle)
    'sway_right': {
        SERVO.R_SHOULDER: 550, SERVO.R_ELBOW: 550,
        SERVO.L_SHOULDER: 450, SERVO.L_ELBOW: 450,
        SERVO.HEAD_YAW: 550,
    },
    'sway_left': {
        SERVO.R_SHOULDER: 450, SERVO.R_ELBOW: 450,
        SERVO.L_SHOULDER: 550, SERVO.L_ELBOW: 550,
        SERVO.HEAD_YAW: 450,
    },

    # EDM moves
    'pump_up': {
        SERVO.R_SHOULDER: 250, SERVO.R_ELBOW: 400,
        SERVO.L_SHOULDER: 750, SERVO.L_ELBOW: 600,
        SERVO.HEAD_PITCH: 350,
    },
    'pump_down': {
        SERVO.R_SHOULDER: 400, SERVO.R_ELBOW: 500,
        SERVO.L_SHOULDER: 600, SERVO.L_ELBOW: 500,
        SERVO.HEAD_PITCH: 550,
    },
    'rave_hands': {
        SERVO.R_SHOULDER: 300, SERVO.R_ELBOW: 350,
        SERVO.L_SHOULDER: 700, SERVO.L_ELBOW: 650,
        SERVO.HEAD_PITCH: 400,
    },
    'fist_pump_r': {
        SERVO.R_SHOULDER: 200, SERVO.R_ELBOW: 300,
        SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 500,
        SERVO.HEAD_YAW: 550, SERVO.HEAD_PITCH: 400,
    },
    'fist_pump_l': {
        SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 500,
        SERVO.L_SHOULDER: 800, SERVO.L_ELBOW: 700,
        SERVO.HEAD_YAW: 450, SERVO.HEAD_PITCH: 400,
    },

    # Hip-hop swagger
    'swagger_r': {
        SERVO.R_SHOULDER: 450, SERVO.R_ELBOW: 450,
        SERVO.L_SHOULDER: 500, SERVO.L_ELBOW: 400,
        SERVO.HEAD_YAW: 550, SERVO.HEAD_PITCH: 520,
    },
    'swagger_l': {
        SERVO.R_SHOULDER: 500, SERVO.R_ELBOW: 400,
        SERVO.L_SHOULDER: 550, SERVO.L_ELBOW: 550,
        SERVO.HEAD_YAW: 450, SERVO.HEAD_PITCH: 520,
    },
}

# Dance sequences
BEAT_SEQUENCES = [
    ['arms_up', 'neutral', 'arms_out', 'neutral'],
    ['right_punch', 'neutral', 'left_punch', 'neutral'],
    ['wave_right', 'wave_left', 'wave_right', 'wave_left'],
    ['head_bob_down', 'head_bob_up', 'head_bob_down', 'head_bob_up'],
    ['celebrate', 'neutral', 'arms_out', 'neutral'],
    ['pump_up', 'pump_down', 'pump_up', 'pump_down'],
    ['sway_right', 'neutral', 'sway_left', 'neutral'],
]

ONSET_MOVES = ['arms_up', 'celebrate', 'right_punch', 'left_punch', 'rave_hands']


# =============================================================================
# SMPL JOINT MAPPING (for retargeting from FACT model)
# =============================================================================

# SMPL joint indices
SMPL_JOINTS = [
    'root', 'l_hip', 'r_hip', 'belly', 'l_knee', 'r_knee',
    'spine', 'l_ankle', 'r_ankle', 'chest', 'l_toes', 'r_toes',
    'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
    'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand'
]

# Mapping from Tony Pro joints to SMPL joints
# Format: {servo_name: {'smpl_joint': str, 'axis': int, 'scale': float}}
RETARGET_MAP = {
    # Head
    'head_pitch': {'servo_id': SERVO.HEAD_PITCH, 'smpl_joint': 'head', 'axis': 0, 'scale': 1.0, 'min': 350, 'max': 650},
    'head_yaw': {'servo_id': SERVO.HEAD_YAW, 'smpl_joint': 'head', 'axis': 1, 'scale': 1.0, 'min': 300, 'max': 700},

    # Right arm (Tony Pro servo directions may need scale adjustment)
    'r_shoulder_pitch': {'servo_id': SERVO.R_SHOULDER, 'smpl_joint': 'r_shoulder', 'axis': 0, 'scale': -1.0, 'min': 200, 'max': 800},
    'r_elbow_pitch': {'servo_id': SERVO.R_ELBOW, 'smpl_joint': 'r_elbow', 'axis': 0, 'scale': 1.0, 'min': 200, 'max': 800},

    # Left arm
    'l_shoulder_pitch': {'servo_id': SERVO.L_SHOULDER, 'smpl_joint': 'l_shoulder', 'axis': 0, 'scale': 1.0, 'min': 200, 'max': 800},
    'l_elbow_pitch': {'servo_id': SERVO.L_ELBOW, 'smpl_joint': 'l_elbow', 'axis': 0, 'scale': -1.0, 'min': 200, 'max': 800},

    # Legs (scaled down for safety)
    'r_hip_roll': {'servo_id': SERVO.R_HIP_ROLL, 'smpl_joint': 'r_hip', 'axis': 2, 'scale': 0.3, 'min': 400, 'max': 600},
    'r_hip_yaw': {'servo_id': SERVO.R_HIP_YAW, 'smpl_joint': 'r_hip', 'axis': 1, 'scale': 0.3, 'min': 400, 'max': 600},
    'r_hip_pitch': {'servo_id': SERVO.R_HIP_PITCH, 'smpl_joint': 'r_hip', 'axis': 0, 'scale': 0.3, 'min': 400, 'max': 600},
    'r_knee_pitch': {'servo_id': SERVO.R_KNEE, 'smpl_joint': 'r_knee', 'axis': 0, 'scale': 0.3, 'min': 400, 'max': 600},
    'r_ankle_pitch': {'servo_id': SERVO.R_ANKLE, 'smpl_joint': 'r_ankle', 'axis': 0, 'scale': 0.3, 'min': 400, 'max': 600},

    'l_hip_roll': {'servo_id': SERVO.L_HIP_ROLL, 'smpl_joint': 'l_hip', 'axis': 2, 'scale': 0.3, 'min': 400, 'max': 600},
    'l_hip_yaw': {'servo_id': SERVO.L_HIP_YAW, 'smpl_joint': 'l_hip', 'axis': 1, 'scale': 0.3, 'min': 400, 'max': 600},
    'l_hip_pitch': {'servo_id': SERVO.L_HIP_PITCH, 'smpl_joint': 'l_hip', 'axis': 0, 'scale': 0.3, 'min': 400, 'max': 600},
    'l_knee_pitch': {'servo_id': SERVO.L_KNEE, 'smpl_joint': 'l_knee', 'axis': 0, 'scale': 0.3, 'min': 400, 'max': 600},
    'l_ankle_pitch': {'servo_id': SERVO.L_ANKLE, 'smpl_joint': 'l_ankle', 'axis': 0, 'scale': 0.3, 'min': 400, 'max': 600},
}

# Active servos for retargeting (start with upper body only for safety)
ACTIVE_RETARGET_JOINTS = [
    'r_shoulder_pitch', 'r_elbow_pitch',
    'l_shoulder_pitch', 'l_elbow_pitch',
    'head_pitch', 'head_yaw',
    # Uncomment after testing stability:
    # 'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee_pitch', 'r_ankle_pitch',
    # 'l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee_pitch', 'l_ankle_pitch',
]


# =============================================================================
# CONTROLLER CLASS
# =============================================================================

class TonyProController:
    """High-level controller for Tony Pro robot."""

    def __init__(self, simulate=False):
        self.simulate = simulate
        self.board = None
        self.controller = None
        self._robot_available = False

        if not simulate:
            try:
                sys.path.insert(0, '/home/pi/TonyPi/HiwonderSDK')
                import hiwonder.ros_robot_controller_sdk as rrc
                from hiwonder.Controller import Controller

                print("Initializing Tony Pro connection...")
                self.board = rrc.Board()
                self.controller = Controller(self.board)
                self._robot_available = True
                print("Tony Pro connected!")
            except ImportError:
                print("HiwonderSDK not found - running in simulation mode")
            except Exception as e:
                print(f"Failed to connect to robot: {e}")
                print("Running in simulation mode")
        else:
            print("Running in simulation mode")

    @property
    def is_connected(self):
        return self._robot_available and self.controller is not None

    def set_servo(self, servo_id, pulse, time_ms=200):
        """Set a single servo position."""
        pulse = clamp_pulse(servo_id, pulse)

        if self.simulate:
            return

        if self.controller:
            self.controller.set_bus_servo_pulse(servo_id, pulse, time_ms)

    def set_servos(self, servo_dict, time_ms=200):
        """Set multiple servos at once.

        Args:
            servo_dict: {servo_id: pulse, ...}
            time_ms: Time to reach positions in milliseconds
        """
        for servo_id, pulse in servo_dict.items():
            self.set_servo(int(servo_id), pulse, time_ms)

    def execute_move(self, move_name, time_ms=200):
        """Execute a named move from MOVES dict."""
        if move_name not in MOVES:
            print(f"Unknown move: {move_name}")
            return False

        move = MOVES[move_name]

        if self.simulate:
            print(f"  -> {move_name}")
            return True

        self.set_servos(move, time_ms)
        return True

    def go_neutral(self, time_ms=500):
        """Move all servos to neutral position."""
        print("Moving to neutral position...")
        self.set_servos(get_neutral(), time_ms)

    def get_servo_position(self, servo_id):
        """Read current position of a servo."""
        if self.controller:
            return self.controller.get_bus_servo_pulse(servo_id)
        return None

    def get_servo_temp(self, servo_id):
        """Read temperature of a servo."""
        if self.controller:
            return self.controller.get_bus_servo_temp(servo_id)
        return None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_config():
    """Load config from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), 'tony_pro_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def print_servo_map():
    """Print the servo mapping in a readable format."""
    print("\nTony Pro Servo Mapping")
    print("=" * 40)
    print(f"{'ID':<4} {'Joint Name':<25} {'Group':<12}")
    print("-" * 40)
    for servo_id in sorted(SERVO_NAMES.keys()):
        name = SERVO_NAMES[servo_id]
        group = "head" if servo_id in HEAD_SERVOS else \
                "right_arm" if servo_id in RIGHT_ARM_SERVOS else \
                "left_arm" if servo_id in LEFT_ARM_SERVOS else \
                "right_leg" if servo_id in RIGHT_LEG_SERVOS else \
                "left_leg"
        print(f"{servo_id:<4} {name:<25} {group:<12}")
    print()


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tony Pro servo configuration')
    parser.add_argument('--map', action='store_true', help='Print servo mapping')
    parser.add_argument('--test', action='store_true', help='Test all moves (simulation)')
    parser.add_argument('--simulate', action='store_true', help='Simulation mode')
    args = parser.parse_args()

    if args.map:
        print_servo_map()

    if args.test:
        import time

        print("\nTesting all dance moves...")
        controller = TonyProController(simulate=True)

        for move_name in MOVES:
            print(f"\nMove: {move_name}")
            controller.execute_move(move_name)
            time.sleep(0.5)

        print("\nDone!")
