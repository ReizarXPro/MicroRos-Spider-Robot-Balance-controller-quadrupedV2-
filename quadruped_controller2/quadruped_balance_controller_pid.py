#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray, String
import numpy as np
import math
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

class ControlMode(Enum):
    FULL_AUTO = "full_auto"
    MIXED_CONTROL = "mixed_control"
    MANUAL_PRIORITY = "manual_priority"
    EMERGENCY_AUTO = "emergency_auto"

@dataclass
class ServoControl:
    manual_enabled: bool = False
    manual_angle: float = 90.0
    balance_weight: float = 1.0  # 0.0 = full manual, 1.0 = full balance
    blend_mode: str = "additive"  # "additive", "weighted", "override"
    last_manual_time: float = 0.0

class AdvancedPIDController:
    """Enhanced PID Controller with adaptive parameters and anti-windup."""
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limit=45.0, 
                 integral_limit=None, derivative_filter=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit or output_limit * 2
        self.derivative_filter = derivative_filter
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_derivative = 0.0
        self.last_time = time.time()
        
        # Adaptive parameters
        self.adaptive_enabled = True
        self.error_history = []
        self.max_history = 20
        
    def update(self, setpoint, current_value):
        """Enhanced PID update with adaptive behavior."""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0.0:
            dt = 0.02
            
        error = setpoint - current_value
        
        # Store error history for adaptive control
        self.error_history.append(abs(error))
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term with filtering
        if dt > 0:
            raw_derivative = (error - self.prev_error) / dt
            derivative = self.derivative_filter * raw_derivative + (1 - self.derivative_filter) * self.prev_derivative
            self.prev_derivative = derivative
        else:
            derivative = 0
            
        d_term = self.kd * derivative
        
        # Adaptive gain adjustment
        if self.adaptive_enabled and len(self.error_history) >= 5:
            avg_error = sum(self.error_history[-5:]) / 5
            if avg_error > 10:  # High error - increase response
                adaptive_gain = 1.2
            elif avg_error < 2:  # Low error - smooth response
                adaptive_gain = 0.8
            else:
                adaptive_gain = 1.0
                
            p_term *= adaptive_gain
            
        output = p_term + i_term + d_term
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        self.prev_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_derivative = 0.0
        self.error_history = []
        self.last_time = time.time()

class QuadrupedBalanceController(Node):
    """
    Advanced Quadruped Balance Controller with sophisticated manual control integration
    """
    
    def __init__(self):
        super().__init__('enhanced_quadruped_balance_controller')
        
        # ROS2 Publishers and Subscribers
        self.imu_subscription = self.create_subscription(
            Float32MultiArray,
            '/esp32/mpu6050/data',
            self.imu_callback,
            10
        )
        
        self.servo_angles_pub = self.create_publisher(
            Int32MultiArray,
            'esp32/servo/angles',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            'quadruped/status',
            10
        )
        
        # Robot configuration
        self.num_servos = 8
        self.servo_names = [
            "Front Left Knee", "Front Left Hip",
            "Front Right Knee", "Front Right Hip", 
            "Rear Left Knee", "Rear Left Hip",
            "Rear Right Knee", "Rear Right Hip"
        ]
        
        # Leg grouping
        self.leg_servos = {
            'FL': [0, 1], 'FR': [2, 3], 'RL': [4, 5], 'RR': [6, 7]
        }
        
        # Servo control objects
        self.servo_controls = [ServoControl() for _ in range(self.num_servos)]
        
        # Base positions for different stances
        self.base_positions = {
            'stand': [180, 90, 0, 90, 0, 90, 180, 90],
            'crouch': [180, 120, 0, 120, 0, 120, 180, 120],
            'wide': [160, 80, 20, 80, 20, 80, 160, 80],
            'narrow': [180, 100, 20, 100, 20, 100, 180, 100]
        }
        self.current_base = 'stand'
        self.current_angles = self.base_positions[self.current_base].copy()
        
        # IMU data
        self.accel_x = self.accel_y = self.accel_z = 0.0
        self.gyro_x = self.gyro_y = self.gyro_z = 0.0
        self.temperature = 0.0
        
        # Calibration offsets
        self.gyro_offset = [0.0, 0.0, 0.0]
        self.accel_offset = [0.0, 0.0, 0.0]
        
        # Calculated angles
        self.roll_angle = 0.0
        self.pitch_angle = 0.0
        self.yaw_rate = 0.0
        
        # Enhanced PID Controllers
        self.roll_pid = AdvancedPIDController(kp=0.5755, ki=0.0144, kd=0.0180, output_limit=40.0)
        self.pitch_pid = AdvancedPIDController(kp=0.5755, ki=0.0072, kd=0.0360, output_limit=40.0)
        
        # Control modes and parameters
        self.control_mode = ControlMode.MIXED_CONTROL
        self.balance_enabled = True
        self.emergency_threshold = 15.0  # degrees
        self.stability_threshold = 8.0  # degrees
        
        # Balance parameters
        self.deadzone = 1.0
        self.max_correction = 50
        self.compensation_gain = 2.0
        self.stability_margin = 5.0
        
        # Complementary filter
        self.alpha = 0.96
        self.dt = 0.02
        
        # Dynamic stability tracking
        self.stability_score = 100.0
        self.fall_risk_level = 0  # 0=safe, 1=caution, 2=warning, 3=emergency
        
        # Manual control timeout
        self.manual_timeout = 2.0  # seconds
        
        # Gait and movement parameters
        self.gait_phase = 0.0
        self.gait_enabled = False
        self.step_height = 15.0
        self.step_frequency = 1.0
        
        # Control loop timer
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        self.status_timer = self.create_timer(0.1, self.publish_status)
        
        # Initialize
        self.publish_servo_angles()
        self.get_logger().info('Enhanced Quadruped Balance Controller initialized')
        
    def imu_callback(self, msg):
        """Process IMU data with enhanced filtering."""
        if len(msg.data) >= 7:
            self.accel_x = msg.data[0] - self.accel_offset[0]
            self.accel_y = msg.data[1] - self.accel_offset[1]
            self.accel_z = msg.data[2] - self.accel_offset[2]
            self.gyro_x = msg.data[3] - self.gyro_offset[0]
            self.gyro_y = msg.data[4] - self.gyro_offset[1]
            self.gyro_z = msg.data[5] - self.gyro_offset[2]
            self.temperature = msg.data[6]
            
            self.calculate_orientation()
            self.update_stability_assessment()
            
    def calculate_orientation(self):
        """Enhanced orientation calculation with noise filtering."""
        # Calculate accelerometer angles
        accel_roll = math.atan2(self.accel_y, self.accel_z) * 180.0 / math.pi
        accel_pitch = math.atan2(-self.accel_x, 
                                math.sqrt(self.accel_y**2 + self.accel_z**2)) * 180.0 / math.pi
        
        # Gyroscope rates
        gyro_roll_rate = self.gyro_x * 180.0 / math.pi
        gyro_pitch_rate = self.gyro_y * 180.0 / math.pi
        self.yaw_rate = self.gyro_z * 180.0 / math.pi
        
        # Adaptive complementary filter
        accel_magnitude = math.sqrt(self.accel_x**2 + self.accel_y**2 + self.accel_z**2)
        accel_confidence = max(0.3, min(1.0, 1.0 - abs(accel_magnitude - 9.81) / 5.0))
        
        adaptive_alpha = self.alpha * accel_confidence
        
        # Apply complementary filter
        self.roll_angle = adaptive_alpha * (self.roll_angle + gyro_roll_rate * self.dt) + \
                         (1 - adaptive_alpha) * accel_roll
        self.pitch_angle = adaptive_alpha * (self.pitch_angle + gyro_pitch_rate * self.dt) + \
                          (1 - adaptive_alpha) * accel_pitch
        
        # Limit angles
        self.roll_angle = np.clip(self.roll_angle, -90, 90)
        self.pitch_angle = np.clip(self.pitch_angle, -90, 90)
        
    def update_stability_assessment(self):
        """Assess robot stability and fall risk."""
        # Calculate stability metrics
        tilt_magnitude = math.sqrt(self.roll_angle**2 + self.pitch_angle**2)
        angular_velocity = math.sqrt(self.gyro_x**2 + self.gyro_y**2 + self.gyro_z**2) * 180.0 / math.pi
        
        # Update stability score (0-100)
        target_stability = max(0, 100 - tilt_magnitude * 2 - angular_velocity)
        self.stability_score = 0.9 * self.stability_score + 0.1 * target_stability
        
        # Determine fall risk level
        if tilt_magnitude > self.emergency_threshold:
            self.fall_risk_level = 3  # Emergency
        elif tilt_magnitude > self.stability_threshold:
            self.fall_risk_level = 2  # Warning
        elif tilt_magnitude > self.stability_threshold * 0.6:
            self.fall_risk_level = 1  # Caution
        else:
            self.fall_risk_level = 0  # Safe
            
        # Auto-switch control mode based on stability
        if self.fall_risk_level >= 3:
            self.control_mode = ControlMode.EMERGENCY_AUTO
        elif self.control_mode == ControlMode.EMERGENCY_AUTO and self.fall_risk_level <= 1:
            self.control_mode = ControlMode.MIXED_CONTROL
            
    def get_leg_control_status(self) -> Dict[str, Dict[str, any]]:
        """Analyze control status for each leg."""
        leg_status = {}
        
        for leg_name, servo_indices in self.leg_servos.items():
            knee_idx, hip_idx = servo_indices
            knee_ctrl = self.servo_controls[knee_idx]
            hip_ctrl = self.servo_controls[hip_idx]
            
            # Determine leg control mode
            if knee_ctrl.manual_enabled or hip_ctrl.manual_enabled:
                if knee_ctrl.manual_enabled and hip_ctrl.manual_enabled:
                    mode = "full_manual"
                else:
                    mode = "partial_manual"
            else:
                mode = "auto"
                
            # Check manual timeout
            current_time = time.time()
            knee_timeout = (current_time - knee_ctrl.last_manual_time) > self.manual_timeout
            hip_timeout = (current_time - hip_ctrl.last_manual_time) > self.manual_timeout
            
            # Calculate effective balance weight for the leg
            leg_balance_weight = (knee_ctrl.balance_weight + hip_ctrl.balance_weight) / 2
            
            leg_status[leg_name] = {
                'mode': mode,
                'balance_weight': leg_balance_weight,
                'knee_manual': knee_ctrl.manual_enabled and not knee_timeout,
                'hip_manual': hip_ctrl.manual_enabled and not hip_timeout,
                'stability_contribution': 1.0 - leg_balance_weight
            }
            
        return leg_status
        
    def calculate_intelligent_balance_correction(self):
        """Advanced balance correction with intelligent servo coordination."""
        corrections = [0.0] * self.num_servos
        
        if not self.balance_enabled:
            return corrections
            
        # Apply deadzone
        roll_input = self.roll_angle if abs(self.roll_angle) > self.deadzone else 0
        pitch_input = self.pitch_angle if abs(self.pitch_angle) > self.deadzone else 0
        
        # Calculate base PID corrections
        roll_correction = self.roll_pid.update(0, roll_input)
        pitch_correction = self.pitch_pid.update(0, pitch_input)
        
        # Get leg control status
        leg_status = self.get_leg_control_status()
        
        # Calculate dynamic correction distribution
        auto_legs = [leg for leg, status in leg_status.items() if status['mode'] == 'auto']
        manual_legs = [leg for leg, status in leg_status.items() if status['mode'] != 'auto']
        
        # Enhance correction for available legs if some legs are manual
        if manual_legs and auto_legs:
            correction_boost = 1.0 + len(manual_legs) * 0.3
            roll_correction *= correction_boost
            pitch_correction *= correction_boost
            
        # Apply corrections to each leg based on its status
        for leg_name, servo_indices in self.leg_servos.items():
            knee_idx, hip_idx = servo_indices
            status = leg_status[leg_name]
            
            # Base corrections based on leg position
            if leg_name in ['FL', 'RL']:  # Left legs
                base_roll = -roll_correction
            else:  # Right legs
                base_roll = roll_correction
                
            if leg_name in ['FL', 'FR']:  # Front legs
                base_pitch = pitch_correction
            else:  # Rear legs
                base_pitch = -pitch_correction
                
            # Calculate leg-specific corrections
            hip_correction = base_roll + base_pitch
            knee_correction = hip_correction * 0.4  # Knee follows hip with reduced magnitude
            
            # Apply corrections based on control mode and weights
            if self.control_mode == ControlMode.EMERGENCY_AUTO:
                # Override manual control in emergency
                corrections[hip_idx] = hip_correction * 1.5
                corrections[knee_idx] = knee_correction * 1.5
                
            elif status['mode'] == 'auto':
                # Full automatic control
                corrections[hip_idx] = hip_correction
                corrections[knee_idx] = knee_correction
                
            elif status['mode'] == 'partial_manual' or status['mode'] == 'full_manual':
                # Apply corrections with balance weights
                hip_ctrl = self.servo_controls[hip_idx]
                knee_ctrl = self.servo_controls[knee_idx]
                
                corrections[hip_idx] = hip_correction * hip_ctrl.balance_weight
                corrections[knee_idx] = knee_correction * knee_ctrl.balance_weight
                
        # Emergency stability enhancement
        if self.fall_risk_level >= 2:
            emergency_multiplier = 1.0 + (self.fall_risk_level - 1) * 0.5
            for i in range(self.num_servos):
                corrections[i] *= emergency_multiplier
                
        # Apply limits
        for i in range(self.num_servos):
            corrections[i] = np.clip(corrections[i], -self.max_correction, self.max_correction)
            
        return corrections
        
    def blend_servo_commands(self, servo_idx: int, balance_correction: float) -> float:
        """Intelligently blend manual and balance commands for a servo."""
        ctrl = self.servo_controls[servo_idx]
        base_angle = self.base_positions[self.current_base][servo_idx]
        
        # Check manual timeout
        current_time = time.time()
        manual_expired = (current_time - ctrl.last_manual_time) > self.manual_timeout
        
        if manual_expired and ctrl.manual_enabled:
            ctrl.manual_enabled = False
            self.get_logger().info(f'Manual control timeout for servo {servo_idx}')
            
        if not ctrl.manual_enabled or self.control_mode == ControlMode.EMERGENCY_AUTO:
            # Pure balance control
            return base_angle + balance_correction
            
        elif ctrl.balance_weight == 0.0:
            # Pure manual control
            return ctrl.manual_angle
            
        else:
            # Blended control
            if ctrl.blend_mode == "additive":
                # Add balance correction to manual position
                return ctrl.manual_angle + balance_correction * ctrl.balance_weight
                
            elif ctrl.blend_mode == "weighted":
                # Weighted average of manual and balance commands
                manual_component = ctrl.manual_angle * (1.0 - ctrl.balance_weight)
                balance_component = (base_angle + balance_correction) * ctrl.balance_weight
                return manual_component + balance_component
                
            else:  # override mode
                if abs(balance_correction) > self.stability_threshold:
                    return base_angle + balance_correction  # Override with balance
                else:
                    return ctrl.manual_angle  # Use manual
                    
    def control_loop(self):
        """Enhanced control loop with intelligent command blending."""
        # Calculate balance corrections
        corrections = self.calculate_intelligent_balance_correction()
        
        # Apply gait if enabled
        if self.gait_enabled:
            self.update_gait()
            
        # Blend commands for each servo
        for i in range(self.num_servos):
            blended_angle = self.blend_servo_commands(i, corrections[i])
            self.current_angles[i] = int(np.clip(blended_angle, 0, 180))
            
        self.publish_servo_angles()
        
    def update_gait(self):
        """Update gait pattern for walking."""
        if not self.gait_enabled:
            return
            
        self.gait_phase += self.step_frequency * self.dt * 2 * math.pi
        if self.gait_phase > 2 * math.pi:
            self.gait_phase -= 2 * math.pi
            
        # Simple trot gait - diagonal legs move together
        for leg_name, servo_indices in self.leg_servos.items():
            knee_idx, hip_idx = servo_indices
            
            # Phase offset for trot gait
            if leg_name in ['FL', 'RR']:
                phase_offset = 0
            else:  # FR, RL
                phase_offset = math.pi
                
            # Calculate gait corrections
            gait_correction = self.step_height * math.sin(self.gait_phase + phase_offset)
            
            # Apply only to knees for basic gait
            if not self.servo_controls[knee_idx].manual_enabled:
                base_knee = self.base_positions[self.current_base][knee_idx]
                self.current_angles[knee_idx] = int(np.clip(base_knee + gait_correction, 0, 180))
                
    def publish_servo_angles(self):
        """Publish servo angles to ESP32."""
        msg = Int32MultiArray()
        msg.data = self.current_angles
        self.servo_angles_pub.publish(msg)
        
    def publish_status(self):
        """Publish detailed status information."""
        leg_status = self.get_leg_control_status()
        
        status_data = {
            'roll': round(self.roll_angle, 2),
            'pitch': round(self.pitch_angle, 2),
            'yaw_rate': round(self.yaw_rate, 2),
            'stability_score': round(self.stability_score, 1),
            'fall_risk': self.fall_risk_level,
            'control_mode': self.control_mode.value,
            'balance_enabled': self.balance_enabled,
            'gait_enabled': self.gait_enabled,
            'temperature': round(self.temperature, 1),
            'leg_status': leg_status
        }
        
        msg = String()
        msg.data = str(status_data)
        self.status_pub.publish(msg)
        
    # --- Start: Added/Modified Public Interface Methods for GUI ---
    
    def set_pid_parameters(self, axis, kp, ki, kd):
        """Set PID parameters for a given axis ('roll' or 'pitch')."""
        if axis == 'roll':
            self.roll_pid.kp = kp
            self.roll_pid.ki = ki
            self.roll_pid.kd = kd
            self.get_logger().info(f"Roll PID updated: Kp={kp}, Ki={ki}, Kd={kd}")
        elif axis == 'pitch':
            self.pitch_pid.kp = kp
            self.pitch_pid.ki = ki
            self.pitch_pid.kd = kd
            self.get_logger().info(f"Pitch PID updated: Kp={kp}, Ki={ki}, Kd={kd}")
        else:
            self.get_logger().warn(f"Invalid PID axis: {axis}")

    def set_compensation_parameters(self, gain, max_compensation):
        """Set balance compensation parameters."""
        self.compensation_gain = gain
        self.max_correction = max_compensation
        self.get_logger().info(f"Compensation updated: Gain={gain}, Max Correction={max_compensation}°")
        
    def set_balance_enabled(self, enabled: bool):
        """Enable or disable the balance system."""
        self.balance_enabled = enabled
        if not enabled:
            # Reset PID controllers when disabling to prevent integral windup
            self.roll_pid.reset()
            self.pitch_pid.reset()
        self.get_logger().info(f"Balance system {'enabled' if enabled else 'disabled'}")
        
    def set_manual_servo_control(self, servo_idx: int, angle: float, 
                               balance_weight: float = 0.0, 
                               blend_mode: str = "additive"):
        """Set manual control for a specific servo with advanced options."""
        if 0 <= servo_idx < self.num_servos:
            ctrl = self.servo_controls[servo_idx]
            ctrl.manual_enabled = True
            ctrl.manual_angle = np.clip(angle, 0, 180)
            ctrl.balance_weight = np.clip(balance_weight, 0.0, 1.0)
            ctrl.blend_mode = blend_mode
            ctrl.last_manual_time = time.time()
            
            self.get_logger().info(
                f'Manual control set for servo {servo_idx} ({self.servo_names[servo_idx]}): '
                f'angle={angle}, weight={balance_weight}, mode={blend_mode}'
            )
            
    def disable_manual_servo_control(self, servo_idx: int):
        """Disable manual control for a specific servo."""
        if 0 <= servo_idx < self.num_servos:
            self.servo_controls[servo_idx].manual_enabled = False
            self.get_logger().info(f'Manual control disabled for servo {servo_idx}')
            
    def set_leg_manual_control(self, leg_name: str, knee_angle: float, hip_angle: float,
                             balance_weight: float = 0.0):
        """Set manual control for an entire leg."""
        if leg_name in self.leg_servos:
            knee_idx, hip_idx = self.leg_servos[leg_name]
            self.set_manual_servo_control(knee_idx, knee_angle, balance_weight)
            self.set_manual_servo_control(hip_idx, hip_angle, balance_weight)
            
    def set_control_mode(self, mode: str):
        """Set overall control mode."""
        try:
            self.control_mode = ControlMode(mode)
            self.get_logger().info(f'Control mode set to: {mode}')
        except ValueError:
            self.get_logger().warn(f'Invalid control mode: {mode}')
            
    def set_base_stance(self, stance: str):
        """Change base stance."""
        if stance in self.base_positions:
            self.current_base = stance
            # When changing stance, disable all manual servo control to reflect the new base pose
            for i in range(self.num_servos):
                self.disable_manual_servo_control(i)
            self.get_logger().info(f'Base stance changed to: {stance}')
        else:
            self.get_logger().warn(f'Unknown stance: {stance}')
            
    def calibrate_sensors(self, samples: int = 200):
        """Enhanced sensor calibration."""
        self.get_logger().info(f'Starting sensor calibration with {samples} samples...')
        
        gyro_sum = [0.0, 0.0, 0.0]
        accel_sum = [0.0, 0.0, 0.0]
        
        # Temporarily store raw values for calibration
        # This assumes imu_callback is running and updating self.accel_*, self.gyro_*
        for _ in range(samples):
            gyro_sum[0] += self.gyro_x + self.gyro_offset[0]
            gyro_sum[1] += self.gyro_y + self.gyro_offset[1] 
            gyro_sum[2] += self.gyro_z + self.gyro_offset[2]
            accel_sum[0] += self.accel_x + self.accel_offset[0]
            accel_sum[1] += self.accel_y + self.accel_offset[1]
            accel_sum[2] += self.accel_z + self.accel_offset[2]
            time.sleep(0.005)
            
        self.gyro_offset = [s / samples for s in gyro_sum]
        self.accel_offset[0] = accel_sum[0] / samples
        self.accel_offset[1] = accel_sum[1] / samples
        # Assuming Z-axis is gravity, we don't want to offset it to zero
        # self.accel_offset[2] = (accel_sum[2] / samples) - 9.81
        
        # Reset PID controllers and orientation after calibration
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.roll_angle = 0.0
        self.pitch_angle = 0.0
        
        self.get_logger().info('Sensor calibration complete')
        
    def enable_gait(self, enable: bool, frequency: float = 1.0, step_height: float = 15.0):
        """Enable/disable gait with parameters."""
        self.gait_enabled = enable
        self.step_frequency = frequency
        self.step_height = step_height
        
        if enable:
            self.get_logger().info(f'Gait enabled: freq={frequency}Hz, height={step_height}°')
        else:
            self.get_logger().info('Gait disabled')
            
    def get_full_status(self) -> Dict:
        """Get comprehensive status information."""
        leg_status = self.get_leg_control_status()
        
        return {
            'orientation': {
                'roll': self.roll_angle,
                'pitch': self.pitch_angle,
                'yaw_rate': self.yaw_rate
            },
            'stability': {
                'score': self.stability_score,
                'fall_risk': self.fall_risk_level,
                'emergency_active': self.control_mode == ControlMode.EMERGENCY_AUTO
            },
            'control': {
                'mode': self.control_mode.value,
                'balance_enabled': self.balance_enabled,
                'gait_enabled': self.gait_enabled,
                'base_stance': self.current_base
            },
            'servos': {
                'angles': self.current_angles.copy(),
                'manual_status': [ctrl.manual_enabled for ctrl in self.servo_controls],
                'balance_weights': [ctrl.balance_weight for ctrl in self.servo_controls]
            },
            'legs': leg_status,
            'sensors': {
                'temperature': self.temperature,
                'accel': [self.accel_x, self.accel_y, self.accel_z],
                'gyro': [self.gyro_x, self.gyro_y, self.gyro_z]
            }
        }
    # --- End: Added/Modified Public Interface Methods for GUI ---


def main(args=None):
    rclpy.init(args=args)
    
    # The GUI script imports and instantiates this class directly.
    # To run this node standalone for testing, you would uncomment the following lines.
    # controller = QuadrupedBalanceController()
    # try:
    #     rclpy.spin(controller)
    # except KeyboardInterrupt:
    #     controller.get_logger().info('Shutting down enhanced balance controller')
    # finally:
    #     controller.destroy_node()
    #     rclpy.shutdown()
    pass # Pass because GUI will handle spinning

if __name__ == '__main__':
    # When running this file directly, it will start a ROS node.
    rclpy.init(args=None)
    controller = QuadrupedBalanceController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down enhanced balance controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()
