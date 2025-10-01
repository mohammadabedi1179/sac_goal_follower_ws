# sac_goal_follower/goal_env.py
import time, math, numpy as np, cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from gymnasium import Env, spaces
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

def _wrap(a):
    while a > math.pi:  a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

class _ROS(Node):
    def __init__(self, cmd_topic, rgb_topic, depth_topic, disp_topic, goal_odom_topic):
        super().__init__('sac_goal_env_node')
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.bridge = CvBridge()
        self.rgb = None; self.depth = None; self.disp = None
        self.fx = None; self.cx = None; self.width = None
        self.goal_pose = None
        # subs
        self.create_subscription(Image, rgb_topic, self._rgb_cb, 10)
        if depth_topic:
            self.create_subscription(Image, depth_topic, self._depth_cb, 10)
        if disp_topic:
            self.create_subscription(Image, disp_topic, self._disp_cb, 10)
        self.create_subscription(CameraInfo, rgb_topic.replace('/image_raw','/camera_info'),
                                 self._cinfo_cb, 10)
        self.create_subscription(Odometry, goal_odom_topic, self._goal_odom_cb, 10)

        # service client for reset
        self.reset_cli = self.create_client(SetEntityState, '/set_entity_state')
        while not self.reset_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/set_entity_state service not available, waiting...')

    def _rgb_cb(self, msg):
        try:
            self.rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.width is None:
                self.width = self.rgb.shape[1]
                if self.cx is None: self.cx = self.width/2.0
        except Exception as e:
            self.get_logger().warn(f"RGB cv2 err: {e}")

    def _depth_cb(self, msg):
        try:
            if msg.encoding == '32FC1':
                self.depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            elif msg.encoding == '16UC1':
                d16 = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                self.depth = d16.astype(np.float32)/1000.0
            else:
                self.depth = self.bridge.imgmsg_to_cv2(msg).astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f"Depth cv2 err: {e}")

    def _disp_cb(self, msg):
        try:
            self.disp = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            self.get_logger().warn(f"Disp cv2 err: {e}")

    def _cinfo_cb(self, msg: CameraInfo):
        if msg.k[0] != 0.0: self.fx = float(msg.k[0])
        if msg.k[2] != 0.0: self.cx = float(msg.k[2])
        if msg.width:       self.width = int(msg.width)

    def _goal_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.goal_pose = (p.x, p.y)

    def send_cmd(self, v, w):
        tw = Twist()
        tw.linear.x = float(v); tw.angular.z = float(w)
        self.cmd_pub.publish(tw)

    def reset_entity(self, name, x, y, z=0.0, yaw=0.0):
        """Teleport an entity in Gazebo."""
        state = EntityState()
        state.name = name
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        qz = math.sin(yaw/2.0)
        qw = math.cos(yaw/2.0)
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        req = SetEntityState.Request()
        req.state = state
        future = self.reset_cli.call_async(req)
        return future  # Return the future for waiting outside

class GoalFollowerEnv(Env):
    """
    Observation: [distance_m, bearing_rad]
    Action: [omega_left, omega_right] (rad/s) → mapped to /cmd_vel using r=0.10, L=0.85
    """
    metadata = {'render_modes': []}

    def __init__(self,
                 cmd_topic='/cmd_vel',
                 rgb_topic='/depth_cam/image_raw',
                 depth_topic='/depth_cam/depth/image_raw',
                 disp_topic='',
                 goal_odom_topic='/goal_marker/odom',
                 wheel_radius=0.10,
                 wheel_separation=0.85,
                 fov_deg=90.0,
                 hsv=(35,85,80,80),      # H:[35,85], S>=80, V>=80
                 min_blob_area=300,
                 dt=0.1,
                 lost_timeout=10.0,
                 success_radius=0.35,
                 time_limit=40.0,
                 c_time=0.01,
                 c_dist=0.5,
                 c_lost=0.1,
                 R_goal=50.0):

        super().__init__()
        self.action_space = spaces.Box(low=np.array([-6.0, -6.0], np.float32),
                                       high=np.array([+6.0, +6.0], np.float32))
        self.observation_space = spaces.Box(low=np.array([0.0, -math.pi], np.float32),
                                            high=np.array([np.inf, math.pi], np.float32))
        # geometry from your robot
        self.r = float(wheel_radius)
        self.L = float(wheel_separation)
        self.fov_deg = float(fov_deg)
        self.h_low, self.h_high, self.s_low, self.v_low = hsv
        self.min_blob_area = int(min_blob_area)
        self.dt = float(dt)
        self.lost_timeout = float(lost_timeout)
        self.success_radius = float(success_radius)
        self.time_limit = float(time_limit)
        self.c_time = float(c_time)
        self.c_dist = float(c_dist)
        self.c_lost = float(c_lost)
        self.R_goal = float(R_goal)

        # ROS init
        rclpy.init(args=None)
        self.ros = _ROS(cmd_topic, rgb_topic, depth_topic, disp_topic, goal_odom_topic)
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(self.ros)

        self._t0 = None
        self._last_seen = None
        self._last_obs_valid = np.array([3.0, 0.0], np.float32)
        self._visible = False
        

    # ---------- helpers ----------
    def _spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.exec.spin_once(timeout_sec=0.01)

    def _detect(self):
        img = self.ros.rgb
        if img is None: return False, None, None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([self.h_low, self.s_low, self.v_low], np.uint8),
            np.array([self.h_high, 255, 255], np.uint8)
        )
        mask = cv2.medianBlur(mask, 5)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return False, None, None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_blob_area: return False, None, None
        M = cv2.moments(c)
        if M['m00'] == 0: return False, None, None
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return True, cx, cy

    def _bearing(self, cx_px):
        if self.ros.fx and self.ros.cx is not None:
            fx = self.ros.fx; cx0 = self.ros.cx
        else:
            w = float(self.ros.width if self.ros.width else 640.0)
            fx = w / (2.0*math.tan(math.radians(self.fov_deg/2.0)))
            cx0 = w / 2.0
        return math.atan2((cx_px - cx0), fx)

    def _distance(self, cx, cy):
        if self.ros.depth is not None:
            if 0 <= cy < self.ros.depth.shape[0] and 0 <= cx < self.ros.depth.shape[1]:
                d = float(self.ros.depth[cy, cx])
                if 0.05 < d < 30.0 and not math.isnan(d):
                    return d
        # optional stereo path
        if self.ros.disp is not None and self.ros.fx:
            B = 0.10  # baseline (adjust!)
            disp = float(self.ros.disp[cy, cx])
            if disp > 0.1:
                return (self.ros.fx * B) / disp
        return None

    def _obs(self, default=False):
        vis, cx, cy = self._detect()
        if vis:
            self._visible = True
            self._last_seen = time.time()
            b = self._bearing(cx)
            d = self._distance(cx, cy)
            if d is None and default:
                d = 3.0
            if d is None:
                return self._last_obs_valid.copy()
            obs = np.array([float(d), float(_wrap(b))], np.float32)
            self._last_obs_valid = obs
            return obs
        else:
            self._visible = False
            return self._last_obs_valid.copy()

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t0 = time.time()
        self._last_seen = time.time()
        self._visible = False
        self.ros.send_cmd(0.0, 0.0)
        
        # teleport robot + marker to start poses
        future_robot = self.ros.reset_entity("my_robot", x=0.0, y=0.0, z=0.3, yaw=0.7854)
        start_time = time.time()
        while rclpy.ok() and not future_robot.done():
            if time.time() - start_time > 2.0:
                self.ros.get_logger().warn("Timeout waiting for robot reset")
                break
            self.exec.spin_once(timeout_sec=0.01)
        if future_robot.done() and future_robot.result() is not None:
            self.ros.get_logger().info(f"Reset my_robot to (0.00, 0.00)")
        else:
            self.ros.get_logger().warn("Failed to reset my_robot")
        
        future_marker = self.ros.reset_entity("goal_marker", x=5.0, y=5.0, z=0.15, yaw=0.0)
        start_time = time.time()
        while rclpy.ok() and not future_marker.done():
            if time.time() - start_time > 2.0:
                self.ros.get_logger().warn("Timeout waiting for goal_marker reset")
                break
            self.exec.spin_once(timeout_sec=0.01)
        if future_marker.done() and future_marker.result() is not None:
            self.ros.get_logger().info(f"Reset goal_marker to (5.00, 5.00)")
        else:
            self.ros.get_logger().warn("Failed to reset goal_marker")
        
        self._spin(0.5)
        obs = self._obs(default=True)
        self.ros.get_logger().info(f"Episode reset: initial obs={obs}")
        return obs, {}

    def step(self, action):
        omega_l = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        omega_r = float(np.clip(action[1], self.action_space.low[1], self.action_space.high[1]))
        v = self.r * (omega_l + omega_r) / 2.0
        w = self.r * (omega_r - omega_l) / self.L
        self.ros.send_cmd(v, w)

        self._spin(self.dt)
        obs = self._obs(default=False)
        dist, bearing = float(obs[0]), float(obs[1])

        reward = 0.0
        reward -= self.c_time * self.dt
        if self._visible and math.isfinite(dist):
            reward -= self.c_dist * dist
        else:
            reward -= self.c_lost * self.dt

        terminated, truncated = False, False
        reason = ""

        if self._visible and math.isfinite(dist) and dist <= self.success_radius:
            reward += self.R_goal
            terminated = True
            reason = "Reached goal"

        if (time.time() - self._last_seen) >= self.lost_timeout:
            terminated = True
            reason = "Lost marker timeout"

        if (time.time() - self._t0) >= self.time_limit:
            truncated = True
            reason = "Time limit reached"

        # logging every step
        self.ros.get_logger().info(
            f"Step: dist={dist:.2f}, bearing={bearing:.2f}, "
            f"visible={self._visible}, reward={reward:.2f}, term={terminated}, trunc={truncated}, reason={reason}"
        )

        return obs, float(reward), terminated, truncated, {"reason": reason}

    def close(self):
        try:
            self.ros.send_cmd(0.0, 0.0)
            self.exec.shutdown()
            self.ros.destroy_node()
        finally:
            rclpy.shutdown()
