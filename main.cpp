#include <Arduino.h>
#include <WiFi.h>
#include <micro_ros_platformio.h>
#include <Wire.h>
#include "Adafruit_PWMServoDriver.h"

//  MPU6050 library
#include "MPU6050.h"

#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/int32_multi_array.h>
#include <std_msgs/msg/float32_multi_array.h>
#include <std_msgs/msg/string.h>

// Servo driver
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);
#define MIN_PULSE_WIDTH 600
#define MAX_PULSE_WIDTH 2600
#define FREQUENCY 50

// Servo configuration
#define NUM_SERVOS 8
const uint8_t SERVO_CHANNELS[NUM_SERVOS] = {0, 1, 2, 3, 4, 5, 6, 7};
int servo_angles[NUM_SERVOS] = {90, 90, 90, 90, 90, 90, 90, 90};

// MPU6050 sensor using default I2C bus
MPU6050 mpu;
bool mpu_initialized = false;
#define MPU_DATA_SIZE 7
float mpu_data[MPU_DATA_SIZE] = {0};

// Micro-ROS objects
rcl_subscription_t servo_array_subscriber;
rcl_publisher_t status_publisher;
rcl_publisher_t mpu_publisher;
std_msgs__msg__Int32MultiArray servo_array_msg;
std_msgs__msg__Float32MultiArray mpu_array_msg;
std_msgs__msg__String status_msg;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;
rcl_timer_t mpu_timer;

// Network configuration
char ssid[] = "YOUR_WIFI_SSID";
char password[] = "YOUR_WIFI_PASSWORD";
IPAddress agent_ip(192, 168, 1, 15); //change it to your pc ip
uint16_t agent_port = 8888;

// Message buffers
char status_buffer[100];
int32_t servo_data_buffer[NUM_SERVOS];
float mpu_data_buffer[MPU_DATA_SIZE];

// Control timing
unsigned long last_wifi_check = 0;
const unsigned long WIFI_CHECK_INTERVAL = 5000;

// Error handling macros
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}

// Error handle loop
void error_loop() {
  Serial.println("Error detected, entering error loop");
  while(1) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(100);
  }
}

// Calculate pulse width from angle
int pulseWidth(int angle) {
  int pulse_wide = map(angle, 0, 180, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  int analog_value = int(float(pulse_wide) / 1000000 * FREQUENCY * 4096);
  return analog_value;
}

// Set servo to specific angle - optimized for minimal latency
void setServoAngle(uint8_t servo_index, int angle) {
  if (servo_index >= NUM_SERVOS) return;
  
  angle = constrain(angle, 0, 180);
  
  if (servo_angles[servo_index] == angle) return;
  
  servo_angles[servo_index] = angle;
  
  pwm.setPWM(SERVO_CHANNELS[servo_index], 0, pulseWidth(angle));
}

// Initialize the MPU6050 sensor
bool initMPU6050() {
  Serial.println("Initializing MPU6050 on default I2C bus...");
  
  mpu.initialize();
  
  bool connected = mpu.testConnection();
  if (connected) {
    Serial.println("MPU6050 connection successful");
    
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
    mpu.setDLPFMode(MPU6050_DLPF_BW_20);
    
    return true;
  } else {
    Serial.println("MPU6050 connection failed");
    return false;
  }
}

// Read MPU6050 data
void readMPU6050Data() {
  if (!mpu_initialized) return;
  
  int16_t ax, ay, az, gx, gy, gz, temp;
  
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  temp = mpu.getTemperature();
  
  mpu_data[0] = ax / 16384.0;
  mpu_data[1] = ay / 16384.0;
  mpu_data[2] = az / 16384.0;
  
  mpu_data[3] = gx / 131.0;
  mpu_data[4] = gy / 131.0;
  mpu_data[5] = gz / 131.0;
  
  mpu_data[6] = temp / 340.0 + 36.53;
}

// Single callback for all servo angles - optimized for speed
void servo_array_callback(const void * msgin) {
  const std_msgs__msg__Int32MultiArray * msg = (const std_msgs__msg__Int32MultiArray *)msgin;
  
  if (msg->data.size < NUM_SERVOS) {
    return;
  }
  
  for (int i = 0; i < NUM_SERVOS; i++) {
    setServoAngle(i, msg->data.data[i]);
  }
}

// Timer callback for periodic status updates
void timer_callback(rcl_timer_t * timer, int64_t last_call_time) {
  RCLC_UNUSED(last_call_time);
  if (timer != NULL) {
    sprintf(status_buffer, "[%d,%d,%d,%d,%d,%d,%d,%d]", 
      servo_angles[0], servo_angles[1], servo_angles[2], servo_angles[3],
      servo_angles[4], servo_angles[5], servo_angles[6], servo_angles[7]);
    status_msg.data.data = status_buffer;
    status_msg.data.size = strlen(status_buffer);
    RCSOFTCHECK(rcl_publish(&status_publisher, &status_msg, NULL));
  }
}

// Timer callback for MPU6050 readings
void mpu_timer_callback(rcl_timer_t * timer, int64_t last_call_time) {
  RCLC_UNUSED(last_call_time);
  if (timer != NULL && mpu_initialized) {
    readMPU6050Data();
    
    for (int i = 0; i < MPU_DATA_SIZE; i++) {
      mpu_array_msg.data.data[i] = mpu_data[i];
    }
    
    RCSOFTCHECK(rcl_publish(&mpu_publisher, &mpu_array_msg, NULL));
  }
}

void setup() {
  Serial.begin(115200);
  
  pwm.begin();
  pwm.setPWMFreq(FREQUENCY);
  
  for (int i = 0; i < NUM_SERVOS; i++) {
    pwm.setPWM(SERVO_CHANNELS[i], 0, pulseWidth(servo_angles[i]));
  }
  
  pinMode(LED_BUILTIN, OUTPUT);
  
  mpu_initialized = initMPU6050();
  
  Serial.println("Connecting to WiFi...");
  
  WiFi.begin(ssid, password);
  
  WiFi.setSleep(false);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  }
  
  digitalWrite(LED_BUILTIN, HIGH);
  
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  
  set_microros_wifi_transports(ssid, password, agent_ip, agent_port);
  
  allocator = rcl_get_default_allocator();
  
  Serial.println("Initializing ROS 2 node...");
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "esp32_servo_controller", "", &support));
  
  servo_array_msg.data.capacity = NUM_SERVOS;
  servo_array_msg.data.size = NUM_SERVOS;
  servo_array_msg.data.data = servo_data_buffer;
  
  mpu_array_msg.data.capacity = MPU_DATA_SIZE;
  mpu_array_msg.data.size = MPU_DATA_SIZE;
  mpu_array_msg.data.data = mpu_data_buffer;
  
  status_msg.data.capacity = sizeof(status_buffer);
  status_msg.data.data = status_buffer;
  
  Serial.println("Creating subscribers...");
  RCCHECK(rclc_subscription_init_default(
    &servo_array_subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32MultiArray),
    "esp32/servo/angles"));
    
  Serial.println("Creating status publisher...");
  RCCHECK(rclc_publisher_init_default(
    &status_publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
    "esp32/servo/status"));
  
  Serial.println("Creating MPU6050 publisher...");
  RCCHECK(rclc_publisher_init_default(
    &mpu_publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
    "esp32/mpu6050/data"));
  
  Serial.println("Creating servo status timer...");
  const unsigned int timer_timeout = 500;
  RCCHECK(rclc_timer_init_default(
    &timer,
    &support,
    RCL_MS_TO_NS(timer_timeout),
    timer_callback));
  
  Serial.println("Creating MPU6050 timer...");
  const unsigned int mpu_timer_timeout = 100;
  RCCHECK(rclc_timer_init_default(
    &mpu_timer,
    &support,
    RCL_MS_TO_NS(mpu_timer_timeout),
    mpu_timer_callback));
  
  Serial.println("Creating executor...");
  RCCHECK(rclc_executor_init(&executor, &support.context, 3, &allocator));
  RCCHECK(rclc_executor_add_subscription(&executor, &servo_array_subscriber, &servo_array_msg, &servo_array_callback, ON_NEW_DATA));
  RCCHECK(rclc_executor_add_timer(&executor, &timer));
  RCCHECK(rclc_executor_add_timer(&executor, &mpu_timer));
  
  sprintf(status_buffer, "ESP32 %d-Servo Controller with MPU6050 started", NUM_SERVOS);
  status_msg.data.size = strlen(status_buffer);
  RCSOFTCHECK(rcl_publish(&status_publisher, &status_msg, NULL));
  
  Serial.println("Micro-ROS setup completed");
}

void loop() {
  unsigned long current_time = millis();
  if (current_time - last_wifi_check > WIFI_CHECK_INTERVAL) {
    last_wifi_check = current_time;
    
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi connection lost. Reconnecting...");
      WiFi.begin(ssid, password);
      
      digitalWrite(LED_BUILTIN, LOW);
      delay(100);
      digitalWrite(LED_BUILTIN, HIGH);
    }
  }
  
  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10)));
}
