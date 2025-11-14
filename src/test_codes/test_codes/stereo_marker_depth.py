#!/usr/bin/env python3
import cv2
import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32, Int32MultiArray
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf2_ros import Buffer, TransformListener

# Try ximgproc (WLS) if available
try:
    import cv2.ximgproc as xip
    _HAS_XIMGPROC = True
except Exception:
    _HAS_XIMGPROC = False

def cam_info_to_KD(ci: CameraInfo):
    K = np.array(ci.k, dtype=np.float64).reshape(3, 3)
    D = np.array(ci.d, dtype=np.float64).reshape(-1, 1)
    size = (ci.width, ci.height)
    return K, D, size

def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float64)

def yaw_from_quat(qx, qy, qz, qw) -> float:
    # yaw around Z
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

class StereoMarkerDepth(Node):
    def __init__(self):
        super().__init__('stereo_marker_depth')

        # ---------- Parameters ----------
        # Topics & frames
        self.declare_parameter('left_image_topic',  '/depth_cam/left/image_raw')
        self.declare_parameter('right_image_topic', '/depth_cam/right/image_raw')
        self.declare_parameter('left_info_topic',   '/depth_cam/left/camera_info')
        self.declare_parameter('right_info_topic',  '/depth_cam/right/camera_info')
        self.declare_parameter('left_optical_frame',  'left_camera_link_optical')
        self.declare_parameter('right_optical_frame', 'right_camera_link_optical')

        # Ground-truth sources
        self.declare_parameter('goal_odom_topic', '/goal_marker/odom')  # nav_msgs/Odometry
        self.declare_parameter('robot_odom_topic','/odom')              # nav_msgs/Odometry
        self.declare_parameter('camera_frame_for_gt','left_camera_link_optical')
        self.declare_parameter('camera_offset_x', 0.4)  # m, forward from robot center (fallback path)

        # Viz
        self.declare_parameter('show_window', True)

        # Red segmentation
        self.declare_parameter('min_area', 600)
        self.declare_parameter('lower_red_1', [0, 120, 80])
        self.declare_parameter('upper_red_1', [12, 255, 255])
        self.declare_parameter('lower_red_2', [170, 120, 80])
        self.declare_parameter('upper_red_2', [180, 255, 255])
        self.declare_parameter('fill_kernel', 7)     # close holes from white dots

        # Robust ROI stats
        self.declare_parameter('inner_roi_ratio', 0.6)
        self.declare_parameter('erode_kernel', 5)
        self.declare_parameter('mad_k', 3.0)
        self.declare_parameter('min_valid_px', 200)
        self.declare_parameter('disp_min', 2.0)
        self.declare_parameter('disp_max', -1.0)     # -1 -> auto

        # Planar fit & smoothing
        self.declare_parameter('use_plane_fit', True)
        self.declare_parameter('smooth_alpha', 0.3)

        # SGBM setup
        self.declare_parameter('num_disparities', 128)   # multiple of 16
        self.declare_parameter('block_size', 7)          # odd
        self.declare_parameter('sgbm_mode', '3WAY')      # SGBM | 3WAY | HH | HH4
        # Compute disparity ONLY on ROI (this node’s point)
        self.declare_parameter('roi_margin', 20)         # pixels margin around box
        # Depth method
        self.declare_parameter('use_Q', True)

        # WLS (if available)
        self.declare_parameter('wls_lambda', 8000.0)
        self.declare_parameter('wls_sigma_color', 1.5)

        # ---------- Setup ----------
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)

        li_topic = self.get_parameter('left_image_topic').value
        ri_topic = self.get_parameter('right_image_topic').value
        linfo_topic = self.get_parameter('left_info_topic').value
        rinfo_topic = self.get_parameter('right_info_topic').value

        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Camera infos
        self.left_info_sub  = self.create_subscription(CameraInfo, linfo_topic,  self.left_info_cb,  qos)
        self.right_info_sub = self.create_subscription(CameraInfo, rinfo_topic, self.right_info_cb, qos)
        self.left_info = None
        self.right_info = None

        # Rectification artifacts
        self.maps_ready = False
        self.left_map1 = self.left_map2 = None
        self.right_map1 = self.right_map2 = None
        self.fx = None
        self.baseline = None
        self.Q = None
        self.size = None

        # Disparity engines
        self.sgbm_L = None
        self.sgbm_R = None
        self.wls = None
        self.num_disp = None

        # Image sync
        self.left_img_sub  = Subscriber(self, Image, li_topic, qos_profile=qos)
        self.right_img_sub = Subscriber(self, Image, ri_topic, qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer([self.left_img_sub, self.right_img_sub], queue_size=20, slop=0.1)
        self.sync.registerCallback(self.image_pair_cb)

        # GT inputs
        self.goal_odom_sub  = self.create_subscription(Odometry, self.get_parameter('goal_odom_topic').value,  self.goal_odom_cb,  10)
        self.robot_odom_sub = self.create_subscription(Odometry, self.get_parameter('robot_odom_topic').value, self.robot_odom_cb, 10)
        self.last_goal_odom: Odometry = None
        self.last_robot_odom: Odometry = None

        # Publishers
        self.dist_pub = self.create_publisher(Float32, 'goal_marker/distance', 10)
        self.roi_pub  = self.create_publisher(Int32MultiArray, 'goal_marker/roi', 10)

        self.depth_ema = None

        self.get_logger().info(f'Listening: {li_topic} & {ri_topic}')
        self.get_logger().info(f'CameraInfo: {linfo_topic} & {rinfo_topic}')
        if not _HAS_XIMGPROC:
            self.get_logger().warn('cv2.ximgproc (WLS) not found — running without WLS.')

    # ---- CameraInfo callbacks ----
    def left_info_cb(self, msg: CameraInfo):
        self.left_info = msg
        self.try_build_maps()

    def right_info_cb(self, msg: CameraInfo):
        self.right_info = msg
        self.try_build_maps()

    # ---- Odometry callbacks for GT ----
    def goal_odom_cb(self, msg: Odometry):
        self.last_goal_odom = msg

    def robot_odom_cb(self, msg: Odometry):
        self.last_robot_odom = msg

    # ---- Build rectification from TF + K,D ----
    def try_build_maps(self):
        if self.maps_ready or self.left_info is None or self.right_info is None:
            return

        Kl, Dl, size_l = cam_info_to_KD(self.left_info)
        Kr, Dr, size_r = cam_info_to_KD(self.right_info)
        if size_l != size_r:
            self.get_logger().error(f'Left/Right sizes differ: {size_l} vs {size_r}')
            return
        self.size = size_l

        left_f  = self.get_parameter('left_optical_frame').value
        right_f = self.get_parameter('right_optical_frame').value
        try:
            tf = self.tf_buffer.lookup_transform(left_f, right_f, Time())
        except Exception as e:
            self.get_logger().warn(f'Waiting TF {left_f}->{right_f}: {e}')
            return

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        tz = tf.transform.translation.z
        q  = tf.transform.rotation
        R_lr = quat_to_rot(q.x, q.y, q.z, q.w)
        T_lr = np.array([[tx], [ty], [tz]], dtype=np.float64)
        self.baseline = abs(float(tx))

        # Stereo rectification
        flags = cv2.CALIB_ZERO_DISPARITY
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            Kl, Dl, Kr, Dr, self.size, R_lr, T_lr, flags=flags, alpha=0
        )
        self.Q = Q
        self.fx = float(P1[0,0])

        # Rectification maps
        self.left_map1, self.left_map2  = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1[:,:3], self.size, cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2[:,:3], self.size, cv2.CV_16SC2)

        # SGBM params
        self.num_disp = int(self.get_parameter('num_disparities').value)
        self.num_disp = max(16, (self.num_disp // 16) * 16)
        block = int(self.get_parameter('block_size').value)
        block = block if block % 2 == 1 else block + 1
        mode_str = str(self.get_parameter('sgbm_mode').value).upper()
        mode_map = {
            'SGBM': cv2.STEREO_SGBM_MODE_SGBM,
            '3WAY': cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            'HH':   cv2.STEREO_SGBM_MODE_HH,
            'HH4':  getattr(cv2, 'STEREO_SGBM_MODE_HH4', cv2.STEREO_SGBM_MODE_HH),
        }
        mode = mode_map.get(mode_str, cv2.STEREO_SGBM_MODE_HH)

        self.sgbm_L = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disp,
            blockSize=block,
            P1=8*block*block,
            P2=32*block*block,
            disp12MaxDiff=1,
            uniquenessRatio=12,
            speckleWindowSize=120,
            speckleRange=2,
            preFilterCap=31,
            mode=mode
        )
        if _HAS_XIMGPROC:
            self.sgbm_R = cv2.StereoSGBM_create(
                minDisparity=-self.num_disp,
                numDisparities=self.num_disp,
                blockSize=block,
                P1=8*block*block,
                P2=32*block*block,
                disp12MaxDiff=1,
                uniquenessRatio=12,
                speckleWindowSize=120,
                speckleRange=2,
                preFilterCap=31,
                mode=mode
            )
            self.wls = xip.createDisparityWLSFilter(self.sgbm_L)
            self.wls.setLambda(float(self.get_parameter('wls_lambda').value))
            self.wls.setSigmaColor(float(self.get_parameter('wls_sigma_color').value))

        self.maps_ready = True
        self.get_logger().info(
            f'Rectified from TF. size={self.size}, fx\'={self.fx:.2f}, baseline={self.baseline:.4f} m, '
            f'WLS={"ON" if _HAS_XIMGPROC else "OFF"} mode={mode_str}'
        )

    # ---- Main callback ----
    def image_pair_cb(self, left_msg: Image, right_msg: Image):
        if not self.maps_ready or self.sgbm_L is None or self.fx is None or (self.baseline is None or self.baseline <= 0):
            return

        left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        # Rectify
        left_rect  = cv2.remap(left,  self.left_map1,  self.left_map2,  cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        left_gray  = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        # ---- Solid red ROI detection (fill holes) ----
        hsv = cv2.cvtColor(left_rect, cv2.COLOR_BGR2HSV)
        lr1 = np.array(self.get_parameter('lower_red_1').value, dtype=np.uint8)
        ur1 = np.array(self.get_parameter('upper_red_1').value, dtype=np.uint8)
        lr2 = np.array(self.get_parameter('lower_red_2').value, dtype=np.uint8)
        ur2 = np.array(self.get_parameter('upper_red_2').value, dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lr1, ur1), cv2.inRange(hsv, lr2, ur2))

        fk = int(self.get_parameter('fill_kernel').value)
        if fk > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fk, fk))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = left_rect.copy()
        if not cnts:
            self._annotate_and_show(vis, np.zeros_like(left_gray, dtype=np.float32), None, None, None, None)
            return

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < int(self.get_parameter('min_area').value):
            self._annotate_and_show(vis, np.zeros_like(left_gray, dtype=np.float32), None, None, None, None)
            return

        # Solidify region & get bounding box
        solid = np.zeros_like(mask)
        cv2.drawContours(solid, [c], -1, 255, thickness=-1)
        mask = solid
        x, y, w, h = cv2.boundingRect(c)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(vis, (int(cx), int(cy)), int(radius), (255,0,0), 2)

        # ---- FULL IMAGE DISPARITY (NO CROP) ----
        if _HAS_XIMGPROC and self.sgbm_R is not None and self.wls is not None:
            dispL = self.sgbm_L.compute(left_gray, right_gray).astype(np.int16)
            dispR = self.sgbm_R.compute(right_gray, left_gray).astype(np.int16)
            disp_full = self.wls.filter(dispL, left_gray, disparity_map_right=dispR).astype(np.float32) / 16.0
        else:
            disp_full = self.sgbm_L.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # ---- ROI sampling (inner central box) ----
        inner = float(np.clip(self.get_parameter('inner_roi_ratio').value, 0.1, 1.0))
        cw = max(1, int(w * inner)); ch = max(1, int(h * inner))
        x_inner = x + (w - cw) // 2
        y_inner = y + (h - ch) // 2

        # Mask for ROI (eroded for robustness)
        roi_mask_full = mask[y:y+h, x:x+w]
        er_k = int(self.get_parameter('erode_kernel').value)
        if er_k > 1:
            kernel = np.ones((er_k, er_k), np.uint8)
            roi_mask_full = cv2.erode(roi_mask_full, kernel, iterations=1)

        # Extract inner ROI from full disparity map
        roi_mask = roi_mask_full[(y_inner - y):(y_inner - y + ch), (x_inner - x):(x_inner - x + cw)]
        roi_disp = disp_full[y_inner:y_inner+ch, x_inner:x_inner+cw]

        disp_min = float(self.get_parameter('disp_min').value)
        disp_max_param = float(self.get_parameter('disp_max').value)
        disp_max = (self.num_disp - 1) if disp_max_param <= 0 else disp_max_param

        valid = (roi_mask > 0) & (roi_disp > disp_min) & (roi_disp < disp_max)
        ys, xs = np.where(valid)
        d_vals = roi_disp[ys, xs]
        min_valid = int(self.get_parameter('min_valid_px').value)

        if d_vals.size < min_valid:
            full_disp_vis = disp_full.copy()
            self._annotate_and_show(vis, full_disp_vis, x, y, w, h, None)
            return

        # Use center of FULL bounding box (not inner ROI)
        cx_full = x + w // 2
        cy_full = y + h // 2

        # Sample disparity at center (robust)
        d_center = disp_full[cy_full, cx_full]
        if d_center <= disp_min or d_center >= disp_max:
            # Fallback to median of valid ROI
            d_center = float(np.median(d_vals)) if d_vals.size > 0 else 0.0

        # Use center disparity for depth
        if d_center > disp_min:
            distance_m = (self.fx * self.baseline) / d_center
        else:
            distance_m = float('nan')

        # EMA smoothing
        alpha = float(self.get_parameter('smooth_alpha').value)
        self.depth_ema = distance_m if self.depth_ema is None else (alpha * distance_m + (1 - alpha) * self.depth_ema)
        distance_display = self.depth_ema

        # Stats for display
        d_med = float(np.median(d_vals))
        mad = 1.4826 * float(np.median(np.abs(d_vals - d_med))) if d_vals.size > 0 else 0.0
        stats = (len(d_vals), mad, d_med, d_center, distance_display, gt)

        # EMA smoothing
        alpha = float(self.get_parameter('smooth_alpha').value)
        self.depth_ema = distance_m if self.depth_ema is None else (alpha * distance_m + (1 - alpha) * self.depth_ema)
        distance_display = self.depth_ema

        # ---- Ground-truth distance ----
        gt = self.compute_ground_truth_distance(left_msg.header.stamp)

        # Publish + annotate & show
        full_disp_vis = disp_full.copy()
        self._annotate_and_show(vis, full_disp_vis, x, y, w, h,
                                (len(d_in), mad, d_med, d_hat, distance_display, gt))
        
    # ---- GT distance helpers ----
    def compute_ground_truth_distance(self, stamp) -> float:
        """
        GT from /goal_marker/odom and /odom by subtracting XY and taking Euclidean norm.
        Assumes both are expressed in the same world-like frame (common in Gazebo).
        Applies a +camera_offset_x (meters) forward along the robot yaw to move from
        robot base to camera position before measuring the distance.
        """
        if self.last_goal_odom is None or self.last_robot_odom is None:
            return float('nan')
        
        # Check timestamp sync
        t_goal = self.last_goal_odom.header.stamp
        t_robot = self.last_robot_odom.header.stamp
        t_now = stamp

        # Convert to seconds
        def to_sec(t): return t.sec + t.nanosec * 1e-9
        if abs(to_sec(t_goal) - to_sec(t_now)) > 0.5 or abs(to_sec(t_robot) - to_sec(t_now)) > 0.5:
            return float('nan')
        
        # goal xy
        gx = self.last_goal_odom.pose.pose.position.x
        gy = self.last_goal_odom.pose.pose.position.y

        # robot xy + camera forward offset in robot heading
        rx = self.last_robot_odom.pose.pose.position.x
        ry = self.last_robot_odom.pose.pose.position.y
        q  = self.last_robot_odom.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        offs = float(self.get_parameter('camera_offset_x').value)  # default 0.4 m
        cam_x = rx + offs * math.cos(yaw)
        cam_y = ry + offs * math.sin(yaw)

        # 2D Euclidean distance (flat ground)
        return float(math.hypot(gx - cam_x, gy - cam_y))


    # ---- annotate & show frame ----
    def _annotate_and_show(self, vis_img, disp_f_full, x=None, y=None, w=None, h=None, stats=None):
        show = bool(self.get_parameter('show_window').value)
        if not show:
            return

        # 1) Draw disparity inset FIRST
        inset_h = inset_w = 0
        try:
            d = np.clip(disp_f_full, 0, self.num_disp if self.num_disp else 128).astype(np.float32)
            d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            d_norm = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
            H, W = vis_img.shape[:2]
            inset_h, inset_w = min(180, H), min(240, W)
            vis_img[0:inset_h, 0:inset_w] = cv2.resize(d_norm, (inset_w, inset_h))
            cv2.rectangle(vis_img, (0,0), (inset_w, inset_h), (200,200,200), 1)
            cv2.putText(vis_img, 'disparity', (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
        except Exception:
            pass

        # 2) Optional ROI box
        if x is not None and y is not None and w is not None and h is not None:
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0,255,0), 2)

        # 3) Draw captions STARTING BELOW the inset
        base_x, base_y = 10, inset_h + 20
        line_h = 24
        if stats is not None:
            N, MAD, d_med, d_hat, dist_est, gt = stats
            self.get_logger().info(f'Ground Truth depth: {gt:.2f}')
            self.get_logger().info(f'Gaol Marker calculated depth: {dist_est:.2f}')
            cv2.putText(vis_img, f'Goal Marker: {dist_est:.2f} m', (base_x, base_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA)
            base_y += line_h
            gt_text = f'GT: {gt:.2f} m' if not math.isnan(gt) else 'GT: n/a'
            cv2.putText(vis_img, gt_text, (base_x, base_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
            base_y += line_h
            cv2.putText(vis_img, f'N={N}  MAD={MAD:.2f}  d_med={d_med:.2f}  d_hat={d_hat:.2f}px',
                        (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

        # 4) Show
        cv2.imshow('Stereo Marker Depth (rectified left)', vis_img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            rclpy.shutdown()
            cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = StereoMarkerDepth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()