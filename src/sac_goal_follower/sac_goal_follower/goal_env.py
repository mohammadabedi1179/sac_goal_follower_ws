# sac_goal_follower/goal_env.py
import time, math, numpy as np, cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from gymnasium import Env, spaces
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import subprocess
from message_filters import Subscriber, ApproximateTimeSynchronizer

def _wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class _ROS(Node):
    def __init__(self, cmd_topic, left_rgb_topic, right_rgb_topic, goal_odom_topic, robot_odom_topic):
        super().__init__("sac_goal_env_node")
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.bridge = CvBridge()
        self.left_rgb = None
        self.right_rgb = None
        self.disparity = None
        self.width = None
        self.goal_pose = None
        self.robot_pose = None
        self.last_left_stamp = None
        self.last_right_stamp = None

        # --- Calibration parameters from left.yaml and right.yaml (new values)
        self.K_left = np.array([[1360.02116, 0, 820.01429],
                                [0, 1360.39793, 615.36189],
                                [0, 0, 1]], np.float64)
        self.D_left = np.array([-0.000154, 0.000171, 0.000019, -0.000004, 0.000000], np.float64)

        self.K_right = np.array([[1359.80732, 0, 820.24415],
                                 [0, 1360.18615, 615.33162],
                                 [0, 0, 1]], np.float64)
        self.D_right = np.array([-0.000661, 0.001202, -0.000040, 0.000068, 0.000000], np.float64)

        # Hard-coded rectification (R1, R2) and projection (P1, P2) from YAML
        self.R1 = np.array([[0.99978141, -0.00025455, -0.02090629],
                            [0.00025495, 0.99999997, 0.00001618],
                            [0.02090629, -0.00002151, 0.99978144]], np.float64)  # From left.yaml rectification_matrix
        self.P1 = np.array([[1415.70791, 0, 826.43549, 0],
                            [0, 1415.70791, 615.33393, 0],
                            [0, 0, 1, 0]], np.float64)  # From left.yaml projection_matrix

        self.R2 = np.array([[0.99989957, -0.00024387, 0.01416982],
                            [0.00024414, 0.99999997, -0.00001712],
                            [-0.01416982, 0.00002058, 0.9998996]], np.float64)  # From right.yaml rectification_matrix
        self.P2 = np.array([[1415.70791, 0, 826.43549, 121.09012],
                            [0, 1415.70791, 615.33393, 0],
                            [0, 0, 1, 0]], np.float64)  # From right.yaml projection_matrix

        self.map_ready = False
        self.map1x = self.map1y = self.map2x = self.map2y = None

        # subs
        self.left_sub = Subscriber(self, Image, left_rgb_topic)  # Now '/depth_cam/left/image_raw'
        self.right_sub = Subscriber(self, Image, right_rgb_topic)  # Now '/depth_cam/right/image_raw'
        self.ts = ApproximateTimeSynchronizer([self.left_sub, self.right_sub],
                                            queue_size=5, slop=0.05)
        self.ts.registerCallback(self._stereo_cb)
        self.create_subscription(Odometry, goal_odom_topic, self._goal_odom_cb, 10)
        self.create_subscription(Odometry, robot_odom_topic, self._robot_odom_cb, 10)

        # service client
        self.reset_cli = self.create_client(SetEntityState, "/set_entity_state")
        for _ in range(10):
            if self.reset_cli.wait_for_service(timeout_sec=1.0):
                break
            self.get_logger().warn("/set_entity_state service not available, waiting...")
        if not self.reset_cli.service_is_ready():
            self.get_logger().error("Failed to connect to /set_entity_state service")

    # === helpers ===
    def _init_rect_maps(self, shape):
        if self.map_ready:
            return
        h, w = shape
        # Use hard-coded R1, P1, R2, P2 instead of computing via stereoRectify
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_left, self.D_left, self.R1, self.P1[:3, :3], (w, h), cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K_right, self.D_right, self.R2, self.P2[:3, :3], (w, h), cv2.CV_32FC1
        )
        self.fx_rect = self.P1[0, 0]
        self.cx_rect = self.P1[0, 2]
        self.cy_rect = self.P1[1, 2]
        self.baseline_rect = -self.P2[0, 3] / self.fx_rect  # Matches ~0.0855 m
        self.map_ready = True
        self.get_logger().info(f"Stereo maps ready: fx={self.fx_rect:.2f}, baseline={self.baseline_rect:.4f} m")

    def _stereo_cb(self, left_msg, right_msg):
        try:
            self.left_rgb = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
            self.right_rgb = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
            self._compute_disparity(self.left_rgb, self.right_rgb)
        except Exception as e:
            self.get_logger().warn(f"Stereo cb err: {e}")


    def _compute_disparity(self, left_img, right_img):
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        self._init_rect_maps(left_gray.shape)

        left_rect = cv2.remap(left_gray, self.map1x, self.map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, self.map2x, self.map2y, cv2.INTER_LINEAR)

        left_rect = cv2.equalizeHist(left_rect)
        right_rect = cv2.equalizeHist(right_rect)

        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 16,
            blockSize=5,
            P1=8 * 5 ** 2,
            P2=32 * 5 ** 2,
            uniquenessRatio=8,
            speckleWindowSize=80,
            speckleRange=16,
            disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        disp = stereo.compute(left_rect, right_rect).astype(np.float32) / 16.0
        disp[disp < 0] = -1.0
        self.disparity = cv2.medianBlur(disp, 5)
        return self.disparity

    def _goal_odom_cb(self, msg):
        p = msg.pose.pose.position
        self.goal_pose = (p.x, p.y)

    def _robot_odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
        self.robot_pose = (p.x, p.y, yaw)

    def send_cmd(self, v, w):
        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.cmd_pub.publish(tw)

    def reset_entity(self, name, x, y, z=0.0, yaw=0.0):
        state = EntityState()
        state.name = name
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        req = SetEntityState.Request()
        req.state = state
        future = self.reset_cli.call_async(req)
        return future


class GoalFollowerEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 cmd_topic="/cmd_vel",
                 left_rgb_topic='/depth_cam/left/image_raw',  # Updated to match remapped topic
                 right_rgb_topic='/depth_cam/right/image_raw',  # Updated to match remapped topic    
                 goal_odom_topic="/goal_marker/odom",
                 robot_odom_topic="/odom",
                 wheel_radius=0.10,
                 wheel_separation=0.35,
                 fov_deg=62.2,
                 min_blob_area=300,
                 dt=0.1,
                 lost_timeout=5.0,
                 success_radius=0.35,
                 time_limit=20.0,
                 c_time=0.01,
                 c_dist=0.1,
                 c_lost=0.1,
                 R_goal=50.0):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([-6.0, -6.0], np.float32),
                                       high=np.array([+6.0, +6.0], np.float32))
        self.observation_space = spaces.Box(low=np.array([0.0, -math.pi], np.float32),
                                            high=np.array([np.inf, math.pi], np.float32))
        self.r = wheel_radius
        self.L = wheel_separation
        self.dt = dt
        self.lost_timeout = lost_timeout
        self.success_radius = success_radius
        self.time_limit = time_limit
        self.c_time = c_time
        self.c_dist = c_dist
        self.c_lost = c_lost
        self.R_goal = R_goal
        self.min_blob_area = min_blob_area

        # nominal (will be overwritten once maps ready)
        self.fx = 1360.89
        self.cx = 820.41
        self.cy = 616.28
        self.baseline = 0.0855

        self.ros = _ROS(cmd_topic, left_rgb_topic, right_rgb_topic, goal_odom_topic, robot_odom_topic)
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(self.ros)

        self._t0 = None
        self._last_seen = None
        self._last_obs_valid = np.array([7.08, 0.0], np.float32)
        self._last_goal_pose = None
        self._visible = False

    def _spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.exec.spin_once(timeout_sec=0.001)

    def _detect(self):
        img = self.ros.left_rgb
        if img is None:
            return False, None, None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.ros.get_logger().info("No red contours detected")
            return False, None, None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_blob_area:
            return False, None, None
        x, y, w, h = cv2.boundingRect(largest)
        cx, cy = x + w // 2, y + h // 2
        self.ros.get_logger().info(f"Detected red marker at ({cx}, {cy}) area={cv2.contourArea(largest)}")
        return True, cx, cy

    def _bearing(self, cx_px):
        fx = self.ros.fx_rect if hasattr(self.ros, "fx_rect") else self.fx
        cx0 = self.ros.cx_rect if hasattr(self.ros, "cx_rect") else self.cx
        return math.atan2((cx_px - cx0), fx)

    def _distance(self, cx, cy):
        if self.ros.disparity is None:
            return 7.08
        H, W = self.ros.disparity.shape
        x0, y0 = int(cx), int(cy)
        x1, y1 = max(0, x0 - 5), max(0, y0 - 5)
        x2, y2 = min(W, x0 + 6), min(H, y0 + 6)
        patch = self.ros.disparity[y1:y2, x1:x2]
        valid = patch[patch > 1.0]
        if valid.size < 10:
            self.ros.get_logger().warn(f"Too few valid disparities near ({cx},{cy})")
            return 7.08
        disp = float(np.median(valid))
        fx = getattr(self.ros, "fx_rect", self.fx)
        baseline = getattr(self.ros, "baseline_rect", self.baseline)
        depth = (baseline * fx) / disp
        if 0.05 < depth < 50.0 and math.isfinite(depth):
            return depth
        self.ros.get_logger().warn(f"Bad depth from disp={disp:.2f}, fallback")
        return 7.08

    def _obs(self, default=False):
        vis, cx, cy = self._detect()
        if vis:
            self._visible = True
            self._last_seen = time.time()
            b = self._bearing(cx)
            d = self._distance(cx, cy)
            if self.ros.robot_pose is not None:
                rx, ry, ryaw = self.ros.robot_pose
                gx = rx + d * math.cos(ryaw + b)
                gy = ry + d * math.sin(ryaw + b)
                self._last_goal_pose = (gx, gy)
            obs = np.array([d, _wrap(b)], np.float32)
            self._last_obs_valid = obs
            return obs
        if self._last_goal_pose and self.ros.robot_pose:
            rx, ry, ryaw = self.ros.robot_pose
            dx = self._last_goal_pose[0] - rx
            dy = self._last_goal_pose[1] - ry
            dist = math.sqrt(dx**2 + dy**2)
            bearing = _wrap(math.atan2(dy, dx) - ryaw)
            obs = np.array([dist, bearing], np.float32)
            self._last_obs_valid = obs
            return obs
        return self._last_obs_valid.copy()

    def _reset_entity_with_retry(self, name, x, y, z=0.0, yaw=0.0, max_attempts=3):
        for attempt in range(max_attempts):
            self.ros.get_logger().info(f"[{name}] Reset attempt {attempt+1}/{max_attempts}")
            qz, qw = math.sin(yaw/2.0), math.cos(yaw/2.0)
            cmd = [
                "ros2", "service", "call", "/set_entity_state",
                "gazebo_msgs/srv/SetEntityState",
                f"{{state: {{name: '{name}', pose: {{position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{z: {qz}, w: {qw}}}}}}}}}"
            ]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if "success: True" in out.stdout or "success=True" in out.stdout:
                    self.ros.get_logger().info(f"[{name}] Reset confirmed by CLI")
                    return True
            except subprocess.TimeoutExpired:
                self.ros.get_logger().warn(f"[{name}] CLI call timed out")
        self.ros.get_logger().error(f"[{name}] Failed to reset after {max_attempts}")
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t0 = time.time()
        self._last_seen = time.time()
        self._visible = False
        self._last_goal_pose = None
        self.ros.send_cmd(0.0, 0.0)
        self._reset_entity_with_retry("my_robot", 0.0, 0.0, 0.3, 0.7854)
        self._reset_entity_with_retry("goal_marker", 5.0, 5.0, 0.75, 0.0)
        self._spin(0.5)
        obs = self._obs(default=True)
        self.ros.get_logger().info(f"Episode reset completed: initial obs={obs}")
        return obs, {}

    def step(self, action):
        omega_l = float(np.clip(action[0], -6.0, 6.0))
        omega_r = float(np.clip(action[1], -6.0, 6.0))
        v = self.r * (omega_l + omega_r) / 2.0
        w = self.r * (omega_r - omega_l) / self.L
        self.ros.send_cmd(np.clip(v, -1.3889, 1.3889), w)
        self._spin(self.dt)
        obs = self._obs()
        dist, bearing = obs
        real_dist = float("inf")
        if self.ros.robot_pose and self.ros.goal_pose:
            rx, ry, _ = self.ros.robot_pose
            gx, gy = self.ros.goal_pose
            real_dist = math.sqrt((gx - rx)**2 + (gy - ry)**2)

        reward = -self.c_time * self.dt
        reward -= self.c_dist * dist if self._visible else self.c_lost * self.dt

        term = trunc = False
        reason = ""
        if self._visible and dist <= self.success_radius:
            reward += self.R_goal
            term = True
            reason = "Reached goal"
        if (time.time() - self._last_seen) >= self.lost_timeout:
            term = True
            reason = "Lost marker timeout"
        if (time.time() - self._t0) >= self.time_limit:
            trunc = True
            reason = "Time limit reached"

        self.ros.get_logger().info(
            f"Step: dist={dist:.2f}, real_dist={real_dist:.2f}, bearing={bearing:.2f}, "
            f"visible={self._visible}, reward={reward:.2f}, term={term}, trunc={trunc}, reason={reason}"
        )
        return obs, float(reward), term, trunc, {"reason": reason}

    def close(self):
        try:
            self.ros.send_cmd(0.0, 0.0)
            self.exec.shutdown()
            self.ros.destroy_node()
        except Exception as e:
            print(f"Error during close: {e}")
        finally:
            if rclpy.ok():
                rclpy.shutdown()