#!/usr/bin/env python3
"""
Tony Pro Servo Configuration and Control Module.

CORRECTED MAPPING (2025-01-17):
- Bus servos 1-8: LEFT side of robot
- Bus servos 9-16: RIGHT side of robot
- PWM servo 1: Head pitch (tilt up/down)
- PWM servo 2: Head yaw (pan left/right)

Usage:
    from tony_pro import SERVO, MOVES, get_neutral, TonyProController
"""

import json
import os
import sys

# =============================================================================
# SERVO ID MAPPING - Tony Pro (CORRECTED)
# =============================================================================

class SERVO:
    """Tony Pro servo IDs by joint name."""

    # Head - PWM servos (NOT bus servos!)
    HEAD_PITCH = 'pwm1'   # Tilt up/down
    HEAD_YAW = 'pwm2'     # Pan left/right

    # Left Arm (bus servos 6-8)
    L_ELBOW = 6           # Elbow pitch
    L_SHOULDER_ROLL = 7   # Shoulder up/down (sideways raise)
    L_SHOULDER_PITCH = 8  # Shoulder forward/backward

    # Right Arm (bus servos 11, 14-16)
    R_ELBOW = 11          # Elbow pitch
    R_SHOULDER_ROLL = 15  # Shoulder up/down (sideways raise)
    R_SHOULDER_PITCH = 16 # Shoulder forward/backward

    # Left Leg (bus servos 1-5)
    L_ANKLE_ROLL = 1      # Foot tilt left/right
    L_ANKLE_PITCH = 2     # Ankle forward/back
    L_KNEE = 3            # Knee bend
    L_HIP_PITCH = 4       # Hip forward/backward
    L_HIP_ROLL = 5        # Hip left/right

    # Right Leg (bus servos 9-10, 12-13)
    R_ANKLE_ROLL = 9      # Foot tilt left/right
    R_ANKLE_PITCH = 10    # Ankle forward/back
    R_HIP_PITCH = 12      # Hip forward/backward
    R_HIP_ROLL = 13       # Hip left/right
    # Note: Right knee might be 14? Need to verify

# Grouped by body part
LEFT_ARM_SERVOS = [SERVO.L_ELBOW, SERVO.L_SHOULDER_ROLL, SERVO.L_SHOULDER_PITCH]
RIGHT_ARM_SERVOS = [SERVO.R_ELBOW, SERVO.R_SHOULDER_ROLL, SERVO.R_SHOULDER_PITCH]
LEFT_LEG_SERVOS = [SERVO.L_ANKLE_ROLL, SERVO.L_ANKLE_PITCH, SERVO.L_KNEE, SERVO.L_HIP_PITCH, SERVO.L_HIP_ROLL]
RIGHT_LEG_SERVOS = [SERVO.R_ANKLE_ROLL, SERVO.R_ANKLE_PITCH, SERVO.R_HIP_PITCH, SERVO.R_HIP_ROLL]

ARM_SERVOS = LEFT_ARM_SERVOS + RIGHT_ARM_SERVOS
LEG_SERVOS = LEFT_LEG_SERVOS + RIGHT_LEG_SERVOS
BUS_SERVOS = list(range(1, 17))

# ID to name mapping (bus servos only)
SERVO_NAMES = {
    1: "left_ankle_roll",
    2: "left_ankle_pitch",
    3: "left_knee_pitch",
    4: "left_hip_pitch",
    5: "left_hip_roll",
    6: "left_elbow_pitch",
    7: "left_shoulder_roll",
    8: "left_shoulder_pitch",
    9: "right_ankle_roll",
    10: "right_ankle_pitch",
    11: "right_elbow_pitch",
    12: "right_hip_pitch",
    13: "right_hip_roll",
    14: "right_knee_pitch",  # Verify this
    15: "right_shoulder_roll",
    16: "right_shoulder_pitch",
}

# Name to ID mapping
SERVO_IDS = {v: k for k, v in SERVO_NAMES.items()}

# =============================================================================
# PULSE VALUES AND RANGES
# =============================================================================

PULSE_CENTER = 500
PULSE_MIN = 0
PULSE_MAX = 1000

# PWM servo settings (head)
PWM_CENTER = 1500
PWM_MIN = 500
PWM_MAX = 2500

# Safe operating ranges per servo
SERVO_LIMITS = {
    # Arms - wider range for dancing
    SERVO.L_ELBOW: (200, 800),
    SERVO.L_SHOULDER_ROLL: (200, 800),
    SERVO.L_SHOULDER_PITCH: (200, 800),
    SERVO.R_ELBOW: (200, 800),
    SERVO.R_SHOULDER_ROLL: (200, 800),
    SERVO.R_SHOULDER_PITCH: (200, 800),

    # Legs - conservative range to prevent falls
    SERVO.L_ANKLE_ROLL: (400, 600),
    SERVO.L_ANKLE_PITCH: (400, 600),
    SERVO.L_KNEE: (400, 600),
    SERVO.L_HIP_PITCH: (400, 600),
    SERVO.L_HIP_ROLL: (400, 600),
    SERVO.R_ANKLE_ROLL: (400, 600),
    SERVO.R_ANKLE_PITCH: (400, 600),
    SERVO.R_HIP_PITCH: (400, 600),
    SERVO.R_HIP_ROLL: (400, 600),
}

# PWM limits for head
PWM_LIMITS = {
    'pwm1': (1200, 1800),  # Head pitch
    'pwm2': (1200, 1800),  # Head yaw
}


def clamp_pulse(servo_id, pulse):
    """Clamp pulse value to safe range for given servo."""
    if isinstance(servo_id, str) and servo_id.startswith('pwm'):
        if servo_id in PWM_LIMITS:
            min_p, max_p = PWM_LIMITS[servo_id]
            return max(min_p, min(max_p, pulse))
        return max(PWM_MIN, min(PWM_MAX, pulse))

    if servo_id in SERVO_LIMITS:
        min_p, max_p = SERVO_LIMITS[servo_id]
        return max(min_p, min(max_p, pulse))
    return max(PULSE_MIN, min(PULSE_MAX, pulse))


def get_neutral():
    """Get neutral pose (all servos at center)."""
    neutral = {servo_id: PULSE_CENTER for servo_id in BUS_SERVOS}
    neutral['pwm1'] = PWM_CENTER
    neutral['pwm2'] = PWM_CENTER
    return neutral


# =============================================================================
# PRE-DEFINED DANCE MOVES
# =============================================================================

MOVES = {
    'neutral': {
        SERVO.L_SHOULDER_PITCH: 500, SERVO.L_SHOULDER_ROLL: 500, SERVO.L_ELBOW: 500,
        SERVO.R_SHOULDER_PITCH: 500, SERVO.R_SHOULDER_ROLL: 500, SERVO.R_ELBOW: 500,
        'pwm1': 1500, 'pwm2': 1500,
    },

    # Arm moves
    'arms_up': {
        SERVO.L_SHOULDER_ROLL: 300, SERVO.L_ELBOW: 500,
        SERVO.R_SHOULDER_ROLL: 700, SERVO.R_ELBOW: 500,
        'pwm1': 1400,
    },
    'arms_out': {
        SERVO.L_SHOULDER_PITCH: 500, SERVO.L_ELBOW: 300,
        SERVO.R_SHOULDER_PITCH: 500, SERVO.R_ELBOW: 700,
    },
    'arms_down': {
        SERVO.L_SHOULDER_ROLL: 700, SERVO.L_ELBOW: 500,
        SERVO.R_SHOULDER_ROLL: 300, SERVO.R_ELBOW: 500,
    },

    # Punch moves
    'right_punch': {
        SERVO.R_SHOULDER_PITCH: 350, SERVO.R_ELBOW: 300,
        SERVO.L_SHOULDER_PITCH: 500, SERVO.L_ELBOW: 500,
        'pwm2': 1600,
    },
    'left_punch': {
        SERVO.L_SHOULDER_PITCH: 650, SERVO.L_ELBOW: 700,
        SERVO.R_SHOULDER_PITCH: 500, SERVO.R_ELBOW: 500,
        'pwm2': 1400,
    },

    # Wave moves
    'wave_right': {
        SERVO.R_SHOULDER_ROLL: 300, SERVO.R_ELBOW: 400,
        SERVO.L_SHOULDER_ROLL: 500, SERVO.L_ELBOW: 500,
        'pwm2': 1550, 'pwm1': 1450,
    },
    'wave_left': {
        SERVO.L_SHOULDER_ROLL: 700, SERVO.L_ELBOW: 600,
        SERVO.R_SHOULDER_ROLL: 500, SERVO.R_ELBOW: 500,
        'pwm2': 1450, 'pwm1': 1450,
    },

    # Head moves
    'head_bob_down': {
        'pwm1': 1600,
    },
    'head_bob_up': {
        'pwm1': 1400,
    },
    'head_left': {
        'pwm2': 1300,
    },
    'head_right': {
        'pwm2': 1700,
    },

    # Celebration
    'celebrate': {
        SERVO.L_SHOULDER_ROLL: 250, SERVO.L_ELBOW: 400,
        SERVO.R_SHOULDER_ROLL: 750, SERVO.R_ELBOW: 600,
        'pwm1': 1350,
    },

    # Sway moves
    'sway_right': {
        SERVO.L_SHOULDER_ROLL: 550, SERVO.L_ELBOW: 550,
        SERVO.R_SHOULDER_ROLL: 450, SERVO.R_ELBOW: 450,
        'pwm2': 1550,
    },
    'sway_left': {
        SERVO.L_SHOULDER_ROLL: 450, SERVO.L_ELBOW: 450,
        SERVO.R_SHOULDER_ROLL: 550, SERVO.R_ELBOW: 550,
        'pwm2': 1450,
    },

    # EDM moves
    'pump_up': {
        SERVO.L_SHOULDER_ROLL: 250, SERVO.L_ELBOW: 400,
        SERVO.R_SHOULDER_ROLL: 750, SERVO.R_ELBOW: 600,
        'pwm1': 1350,
    },
    'pump_down': {
        SERVO.L_SHOULDER_ROLL: 400, SERVO.L_ELBOW: 500,
        SERVO.R_SHOULDER_ROLL: 600, SERVO.R_ELBOW: 500,
        'pwm1': 1600,
    },
    'rave_hands': {
        SERVO.L_SHOULDER_ROLL: 300, SERVO.L_ELBOW: 350,
        SERVO.R_SHOULDER_ROLL: 700, SERVO.R_ELBOW: 650,
        'pwm1': 1400,
    },
    'fist_pump_r': {
        SERVO.R_SHOULDER_ROLL: 200, SERVO.R_ELBOW: 300,
        SERVO.L_SHOULDER_ROLL: 500, SERVO.L_ELBOW: 500,
        'pwm2': 1550, 'pwm1': 1400,
    },
    'fist_pump_l': {
        SERVO.L_SHOULDER_ROLL: 800, SERVO.L_ELBOW: 700,
        SERVO.R_SHOULDER_ROLL: 500, SERVO.R_ELBOW: 500,
        'pwm2': 1450, 'pwm1': 1400,
    },

    # Hip-hop swagger
    'swagger_r': {
        SERVO.L_SHOULDER_ROLL: 450, SERVO.L_ELBOW: 450,
        SERVO.R_SHOULDER_ROLL: 500, SERVO.R_ELBOW: 400,
        'pwm2': 1550, 'pwm1': 1520,
    },
    'swagger_l': {
        SERVO.L_SHOULDER_ROLL: 500, SERVO.L_ELBOW: 400,
        SERVO.R_SHOULDER_ROLL: 550, SERVO.R_ELBOW: 550,
        'pwm2': 1450, 'pwm1': 1520,
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

SMPL_JOINTS = [
    'root', 'l_hip', 'r_hip', 'belly', 'l_knee', 'r_knee',
    'spine', 'l_ankle', 'r_ankle', 'chest', 'l_toes', 'r_toes',
    'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
    'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand'
]

# Mapping from Tony Pro joints to SMPL joints
RETARGET_MAP = {
    # Head (PWM servos - full FACT output - no restrictions)
    'head_pitch': {'servo_id': 'pwm1', 'smpl_joint': 'head', 'axis': 0, 'scale': 1.0, 'min': 500, 'max': 2500, 'center': 1500},
    'head_yaw': {'servo_id': 'pwm2', 'smpl_joint': 'head', 'axis': 1, 'scale': 1.0, 'min': 500, 'max': 2500, 'center': 1500},

    # Left arm (full FACT output - no restrictions)
    'l_shoulder_pitch': {'servo_id': SERVO.L_SHOULDER_PITCH, 'smpl_joint': 'l_shoulder', 'axis': 0, 'scale': 1.0, 'min': 0, 'max': 1000, 'center': 500},
    'l_shoulder_roll': {'servo_id': SERVO.L_SHOULDER_ROLL, 'smpl_joint': 'l_shoulder', 'axis': 2, 'scale': 1.0, 'min': 0, 'max': 1000, 'center': 500},
    'l_elbow_pitch': {'servo_id': SERVO.L_ELBOW, 'smpl_joint': 'l_elbow', 'axis': 0, 'scale': -1.0, 'min': 0, 'max': 1000, 'center': 500},

    # Right arm (full FACT output - no restrictions)
    'r_shoulder_pitch': {'servo_id': SERVO.R_SHOULDER_PITCH, 'smpl_joint': 'r_shoulder', 'axis': 0, 'scale': -1.0, 'min': 0, 'max': 1000, 'center': 500},
    'r_shoulder_roll': {'servo_id': SERVO.R_SHOULDER_ROLL, 'smpl_joint': 'r_shoulder', 'axis': 2, 'scale': -1.0, 'min': 0, 'max': 1000, 'center': 500},
    'r_elbow_pitch': {'servo_id': SERVO.R_ELBOW, 'smpl_joint': 'r_elbow', 'axis': 0, 'scale': 1.0, 'min': 0, 'max': 1000, 'center': 500},

    # Left leg (full FACT output - no restrictions)
    'l_hip_pitch': {'servo_id': SERVO.L_HIP_PITCH, 'smpl_joint': 'l_hip', 'axis': 0, 'scale': 1.0, 'min': 0, 'max': 1000, 'center': 500},
    'l_hip_roll': {'servo_id': SERVO.L_HIP_ROLL, 'smpl_joint': 'l_hip', 'axis': 2, 'scale': 1.0, 'min': 0, 'max': 1000, 'center': 500},
    'l_knee_pitch': {'servo_id': SERVO.L_KNEE, 'smpl_joint': 'l_knee', 'axis': 0, 'scale': 1.0, 'min': 0, 'max': 1000, 'center': 500},
    'l_ankle_pitch': {'servo_id': SERVO.L_ANKLE_PITCH, 'smpl_joint': 'l_ankle', 'axis': 0, 'scale': 1.0, 'min': 0, 'max': 1000, 'center': 500},
    'l_ankle_roll': {'servo_id': SERVO.L_ANKLE_ROLL, 'smpl_joint': 'l_ankle', 'axis': 2, 'scale': 1.0, 'min': 0, 'max': 1000, 'center': 500},

    # Right leg (full FACT output - no restrictions)
    'r_hip_pitch': {'servo_id': SERVO.R_HIP_PITCH, 'smpl_joint': 'r_hip', 'axis': 0, 'scale': -1.0, 'min': 0, 'max': 1000, 'center': 500},
    'r_hip_roll': {'servo_id': SERVO.R_HIP_ROLL, 'smpl_joint': 'r_hip', 'axis': 2, 'scale': -1.0, 'min': 0, 'max': 1000, 'center': 500},
    'r_knee_pitch': {'servo_id': 14, 'smpl_joint': 'r_knee', 'axis': 0, 'scale': -1.0, 'min': 0, 'max': 1000, 'center': 500},
    'r_ankle_pitch': {'servo_id': SERVO.R_ANKLE_PITCH, 'smpl_joint': 'r_ankle', 'axis': 0, 'scale': -1.0, 'min': 0, 'max': 1000, 'center': 500},
    'r_ankle_roll': {'servo_id': SERVO.R_ANKLE_ROLL, 'smpl_joint': 'r_ankle', 'axis': 2, 'scale': -1.0, 'min': 0, 'max': 1000, 'center': 500},
}

# Active servos for retargeting (full body with safety scaling)
ACTIVE_RETARGET_JOINTS = [
    # Head
    'head_pitch', 'head_yaw',
    # Arms (full range)
    'l_shoulder_pitch', 'l_shoulder_roll', 'l_elbow_pitch',
    'r_shoulder_pitch', 'r_shoulder_roll', 'r_elbow_pitch',
    # Legs (conservative scaling to prevent falls)
    'l_hip_pitch', 'l_hip_roll', 'l_knee_pitch', 'l_ankle_pitch', 'l_ankle_roll',
    'r_hip_pitch', 'r_hip_roll', 'r_knee_pitch', 'r_ankle_pitch', 'r_ankle_roll',
]


# =============================================================================
# CONTROLLER CLASS
# =============================================================================

class TonyProController:
    """High-level controller for Tony Pro robot."""

    def __init__(self, simulate=False):
        self.simulate = simulate
        self.board = None
        self._robot_available = False

        if not simulate:
            try:
                sys.path.insert(0, '/home/pi/TonyPi/HiwonderSDK')
                import hiwonder.ros_robot_controller_sdk as rrc

                print("Initializing Tony Pro connection...")
                self.board = rrc.Board()
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
        return self._robot_available and self.board is not None

    def set_servo(self, servo_id, pulse, time_ms=200):
        """Set a single servo position."""
        pulse = clamp_pulse(servo_id, pulse)

        if self.simulate:
            return

        if self.board:
            if isinstance(servo_id, str) and servo_id.startswith('pwm'):
                # PWM servo (head)
                pwm_id = int(servo_id[3:])
                self.board.pwm_servo_set_position(time_ms / 1000.0, ((pwm_id, int(pulse)),))
            else:
                # Bus servo (body)
                self.board.bus_servo_set_position(time_ms / 1000.0, ((int(servo_id), int(pulse)),))

    def set_servos(self, servo_dict, time_ms=200):
        """Set multiple servos at once."""
        bus_cmds = []
        pwm_cmds = []

        for servo_id, pulse in servo_dict.items():
            pulse = clamp_pulse(servo_id, pulse)
            if isinstance(servo_id, str) and servo_id.startswith('pwm'):
                pwm_id = int(servo_id[3:])
                pwm_cmds.append((pwm_id, int(pulse)))
            else:
                bus_cmds.append((int(servo_id), int(pulse)))

        if self.simulate:
            return

        if self.board:
            if bus_cmds:
                self.board.bus_servo_set_position(time_ms / 1000.0, tuple(bus_cmds))
            if pwm_cmds:
                self.board.pwm_servo_set_position(time_ms / 1000.0, tuple(pwm_cmds))

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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_servo_map():
    """Print the servo mapping in a readable format."""
    print("\nTony Pro Servo Mapping (CORRECTED)")
    print("=" * 50)
    print("BUS SERVOS (1-16):")
    print(f"{'ID':<4} {'Joint Name':<25} {'Side':<10}")
    print("-" * 40)
    for servo_id in sorted(SERVO_NAMES.keys()):
        name = SERVO_NAMES[servo_id]
        side = "LEFT" if servo_id <= 8 else "RIGHT"
        print(f"{servo_id:<4} {name:<25} {side:<10}")

    print("\nPWM SERVOS (Head):")
    print("  PWM 1: head_pitch (tilt up/down)")
    print("  PWM 2: head_yaw (pan left/right)")
    print()


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tony Pro servo configuration')
    parser.add_argument('--map', action='store_true', help='Print servo mapping')
    parser.add_argument('--test', action='store_true', help='Test all moves')
    parser.add_argument('--simulate', action='store_true', help='Simulation mode')
    args = parser.parse_args()

    if args.map:
        print_servo_map()

    if args.test:
        import time

        print("\nTesting all dance moves...")
        controller = TonyProController(simulate=args.simulate)

        for move_name in MOVES:
            print(f"\nMove: {move_name}")
            controller.execute_move(move_name)
            time.sleep(0.5)

        print("\nDone!")
