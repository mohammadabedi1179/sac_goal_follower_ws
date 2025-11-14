#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import os


def is_identity(R, tol=1e-6):
    return np.allclose(R, np.eye(3), atol=tol)


def is_zero_dist(D, tol=1e-9):
    return (D is None) or np.allclose(D, 0.0, atol=tol)


class DisparityDebug(Node):
    def __init__(self):
        super().__init__('disparity_debug')
        self.bridge = CvBridge()

        # Latest frames / maps / camera model
        self.left_rgb = None
        self.right_rgb = None
        self.left_rect_bgr = None
        self.disparity = None

        # CameraInfo cache (live)
        self.left_info = None
        self.right_info = None
        self.fx_rect = None
        self.cx_rect = None
        self.cy_rect = None
        self.baseline_rect = None

        # Rectification maps (built only if needed)
        self.map_ready = False
        self.map1x = self.map1y = self.map2x = self.map2y = None
        self.maps_shape = None  # (h, w)

        # Output dir
        self.save_dir = os.path.expanduser('~/disparity_debug')
        os.makedirs(self.save_dir, exist_ok=True)

        # --- Subscribers ---
        self.left_sub_img = Subscriber(self, Image, '/depth_cam/left/image_raw')
        self.right_sub_img = Subscriber(self, Image, '/depth_cam/right/image_raw')
        self.ts = ApproximateTimeSynchronizer([self.left_sub_img, self.right_sub_img],
                                              queue_size=5, slop=0.05)
        self.ts.registerCallback(self._stereo_cb)

        # CameraInfo subscriptions (plain rclpy; arrive asynchronously)
        self.create_subscription(CameraInfo, '/depth_cam/left/camera_info',
                                 self._left_info_cb, 10)
        self.create_subscription(CameraInfo, '/depth_cam/right/camera_info',
                                 self._right_info_cb, 10)

        self.get_logger().info('DisparityDebug node initialized')

    # ===================== CameraInfo ===================== #
    def _left_info_cb(self, msg: CameraInfo):
        self.left_info = msg
        self._maybe_update_model()

    def _right_info_cb(self, msg: CameraInfo):
        self.right_info = msg
        self._maybe_update_model()

    def _maybe_update_model(self):
        """Set fx/cx/cy/baseline from live CameraInfo and reset maps if size changes."""
        if self.left_info is None or self.right_info is None:
            return

        # Intrinsics from left (rectified)
        P_left = np.array(self.left_info.p, dtype=np.float64).reshape(3, 4)
        P_right = np.array(self.right_info.p, dtype=np.float64).reshape(3, 4)

        fx = float(P_left[0, 0])  # rectified fx (left/right should match)
        cx = float(P_left[0, 2])
        cy = float(P_left[1, 2])

        # Baseline from the camera that has nonzero Tx in P
        tx_r = float(P_right[0, 3])
        tx_l = float(P_left[0, 3])
        tx = tx_r if abs(tx_r) > 1e-6 else tx_l  # pixels
        baseline = abs(tx) / fx  # meters
        baseline = 0.072

        self.fx_rect, self.cx_rect, self.cy_rect, self.baseline_rect = fx, cx, cy, baseline

        # Reset maps if needed (size and rectification status can change)
        self.map_ready = False
        self.maps_shape = None

        self.get_logger().info(
            f'Camera model set from CameraInfo: fx={fx:.2f}, cx={cx:.2f}, cy={cy:.2f}, B={baseline:.4f} m'
        )

    # ===================== Rectification ===================== #
    def _init_rect_maps(self, shape_hw):
        """Build undistort/rectify maps only if the pair is NOT already rectified."""
        if self.map_ready and self.maps_shape == tuple(shape_hw):
            return
        if self.left_info is None or self.right_info is None:
            return  # will try again next frame

        h, w = shape_hw
        self.maps_shape = (h, w)

        # Parse K/D/R/P from CameraInfo (live)
        K1 = np.array(self.left_info.k, dtype=np.float64).reshape(3, 3)
        D1 = np.array(self.left_info.d, dtype=np.float64).reshape(-1)
        R1 = np.array(self.left_info.r, dtype=np.float64).reshape(3, 3)
        P1 = np.array(self.left_info.p, dtype=np.float64).reshape(3, 4)

        K2 = np.array(self.right_info.k, dtype=np.float64).reshape(3, 3)
        D2 = np.array(self.right_info.d, dtype=np.float64).reshape(-1)
        R2 = np.array(self.right_info.r, dtype=np.float64).reshape(3, 3)
        P2 = np.array(self.right_info.p, dtype=np.float64).reshape(3, 4)

        # If already rectified (typical in Gazebo): D≈0 and R≈I
        if is_zero_dist(D1) and is_zero_dist(D2) and is_identity(R1) and is_identity(R2):
            # No maps needed
            self.map1x = self.map1y = self.map2x = self.map2y = None
            self.map_ready = True
            self.get_logger().info('Input appears already rectified (D≈0, R≈I); skipping remap.')
            return

        # Otherwise build remap using live K/D/R and P[:3, :3]
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1[:3, :3], (w, h), cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2[:3, :3], (w, h), cv2.CV_32FC1
        )
        self.map_ready = True
        self.get_logger().info('Built undistort/rectify maps from live CameraInfo.')

    # ===================== Disparity ===================== #
    """def _compute_disparity(self, left_bgr, right_bgr):
        if self.fx_rect is None or self.baseline_rect is None:
            self.get_logger().warn('Camera model not ready yet; skipping frame.')
            return None

        self._init_rect_maps(left_bgr.shape[:2])

        # Use images as-is if already rectified; else remap
        if self.map1x is None:
            self.left_rect_bgr = left_bgr.copy()
            right_rect_bgr = right_bgr
        else:
            self.left_rect_bgr = cv2.remap(left_bgr, self.map1x, self.map1y, cv2.INTER_LINEAR)
            right_rect_bgr = cv2.remap(right_bgr, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # Save color rectified (for debugging & detection)
        cv2.imwrite(os.path.join(self.save_dir, 'left_rect_color.png'), self.left_rect_bgr)

        # Grayscale for SGBM
        left_rect = cv2.cvtColor(self.left_rect_bgr, cv2.COLOR_BGR2GRAY)
        right_rect = cv2.cvtColor(right_rect_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(self.save_dir, 'left_rect_gray.png'), left_rect)
        cv2.imwrite(os.path.join(self.save_dir, 'right_rect_gray.png'), right_rect)

        # Equalize to boost contrast
        left_rect = cv2.equalizeHist(left_rect)
        right_rect = cv2.equalizeHist(right_rect)

        # SGBM settings: balanced to keep matches on low-texture while limiting speckle
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 12,   # 192
            blockSize=26,
            P1=8 * 9 * 9,
            P2=32 * 9 * 9,
            uniquenessRatio=7,
            speckleWindowSize=80,
            speckleRange=2,
            disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        disp = stereo.compute(left_rect, right_rect).astype(np.float32) / 16.0
        disp[disp < 0] = -1.0
        self.disparity = cv2.medianBlur(disp, 5)

        # Save disparity visualization
        dv = np.where(self.disparity < 0, 0, self.disparity)
        dv = cv2.normalize(dv, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        dv = cv2.applyColorMap(dv, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(self.save_dir, 'disp_vis.png'), dv)
        return self.disparity"""
    def _compute_disparity(self, left_bgr, right_bgr):
        if self.fx_rect is None or self.baseline_rect is None:
            self.get_logger().warn('Camera model not ready yet; skipping frame.')
            return None
        self._init_rect_maps(left_bgr.shape[:2])

        # Use images as-is if already rectified; else remap
        if self.map1x is None:
            self.left_rect_bgr = left_bgr.copy()
            right_rect_bgr = right_bgr
        else:
            self.left_rect_bgr = cv2.remap(left_bgr, self.map1x, self.map1y, cv2.INTER_LINEAR)
            right_rect_bgr = cv2.remap(right_bgr, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # Save color rectified (for debugging & detection)
        cv2.imwrite(os.path.join(self.save_dir, 'left_rect_color.png'), self.left_rect_bgr)

        # Grayscale for SGBM
        left_rect = cv2.cvtColor(self.left_rect_bgr, cv2.COLOR_BGR2GRAY)
        right_rect = cv2.cvtColor(right_rect_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(self.save_dir, 'left_rect_gray.png'), left_rect)
        cv2.imwrite(os.path.join(self.save_dir, 'right_rect_gray.png'), right_rect)

        # --- MODIFICATIONS START HERE ---

        # 1. REMOVED: Histogram equalization is likely amplifying noise in this textureless scene.
        # left_rect = cv2.equalizeHist(left_rect)
        # right_rect = cv2.equalizeHist(right_rect)

        # 2. UPDATED: SGBM settings tuned for low-texture scenes.
        block_size = 11  # Increased block size
        min_disp = 0
        num_disp = 16 * 4 # 128 disparities. Can be increased if objects are very close.
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,     # Increased P1 for more smoothness
            P2=32 * 3 * block_size**2,    # Increased P2 for more smoothness
            uniquenessRatio=10,           # More strict uniqueness check
            speckleWindowSize=150,        # Increased speckle filtering
            speckleRange=2,
            disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_HH, # Using the more robust HH mode
        )
        
        disp = stereo.compute(left_rect, right_rect).astype(np.float32) / 16.0
        
        # --- MODIFICATIONS END HERE ---
        
        disp[disp < 0] = -1.0
        self.disparity = cv2.medianBlur(disp, 5) # Median blur is still a good final touch

        # Save disparity visualization
        dv = np.where(self.disparity < 0, 0, self.disparity)
        dv = cv2.normalize(dv, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        dv = cv2.applyColorMap(dv, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(self.save_dir, 'disp_vis.png'), dv)
        
        return self.disparity

    # ===================== Detection (HSV ∪ Lab-a*) ===================== #
    def _detect_on_rectified(self):
        """
        Prefer robust color-based (HSV ∪ Lab-a*) detection.
        If too small, fall back to texture-based edges in the lower 2/3.
        """
        if self.left_rect_bgr is None:
            self.get_logger().warn('No rectified color image for detection')
            return False, (int(self.cx_rect or 0), int(self.cy_rect or 0)), None, None

        H, W = self.left_rect_bgr.shape[:2]
        area_min = max(50, int(0.0003 * H * W))

        # HSV red (wide)
        hsv = cv2.cvtColor(self.left_rect_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, (0, 20, 20), (10, 255, 255)) | \
                   cv2.inRange(hsv, (170, 20, 20), (180, 255, 255))

        # Lab a* (red-green axis), with Otsu on bottom 2/3 ROI
        lab = cv2.cvtColor(self.left_rect_bgr, cv2.COLOR_BGR2Lab)
        a = lab[:, :, 1]
        roi = a[int(H * 0.33):, :]
        otsu_val, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_val = max(128, int(otsu_val))
        mask_lab = (a >= thr_val).astype(np.uint8) * 255

        mask = cv2.bitwise_or(mask_hsv, mask_lab)
        k3 = np.ones((3, 3), np.uint8)
        k7 = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= area_min:
                filled = np.zeros_like(mask)
                cv2.drawContours(filled, [largest], -1, 255, thickness=-1)
                inner = cv2.erode(filled, k7, iterations=1)
                M = cv2.moments(inner)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    vis = self.left_rect_bgr.copy()
                    cv2.drawContours(vis, [largest], -1, (0, 255, 0), 2)
                    cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
                    cv2.imwrite(os.path.join(self.save_dir, 'mask_vis.png'), vis)
                    self.get_logger().info(f'Goal marker (color) at ({cx},{cy})')
                    return True, (cx, cy), inner, largest

        # Fallback: texture/luma edges in lower 2/3
        gray = cv2.cvtColor(self.left_rect_bgr, cv2.COLOR_BGR2GRAY)
        roi_mask = np.zeros_like(gray); roi_mask[int(H * 0.33):, :] = 255
        gray_roi = cv2.bitwise_and(gray, gray, mask=roi_mask)
        edges = cv2.Canny(gray_roi, 40, 120)
        edges = cv2.dilate(edges, k3, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().warn('No goal marker detected (both methods)')
            return False, (int(self.cx_rect or 0), int(self.cy_rect or 0)), None, None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < area_min:
            self.get_logger().warn('Fallback contour too small')
            return False, (int(self.cx_rect or 0), int(self.cy_rect or 0)), None, None

        filled = np.zeros_like(gray)
        cv2.drawContours(filled, [largest], -1, 255, thickness=-1)
        inner = cv2.erode(filled, k7, iterations=1)
        M = cv2.moments(inner)
        if M['m00'] <= 0:
            self.get_logger().warn('Zero moment in fallback mask')
            return False, (int(self.cx_rect or 0), int(self.cy_rect or 0)), None, None

        cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
        vis = self.left_rect_bgr.copy()
        cv2.drawContours(vis, [largest], -1, (0, 200, 255), 2)
        cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
        cv2.imwrite(os.path.join(self.save_dir, 'mask_vis.png'), vis)
        self.get_logger().info(f'Goal marker (fallback) at ({cx},{cy})')
        return True, (cx, cy), inner, largest

    # ===================== Depth ===================== #
    def _depth_from_mask(self, mask_eroded):
        if self.disparity is None:
            self.get_logger().warn('No disparity map available')
            return 7.08
        if mask_eroded is None or mask_eroded.sum() == 0:
            self.get_logger().warn('No valid mask; returning default')
            return 7.08

        vals = self.disparity[(mask_eroded > 0) & (self.disparity > 1.5)]
        if vals.size < 15:
            self.get_logger().warn(f'Too few valid disparities inside mask ({vals.size})')
            return 7.08

        disp = float(np.median(vals))
        depth = (self.fx_rect * self.baseline_rect) / disp
        self.get_logger().info(
            f'Raw depth: {depth:.2f} m, disparity={disp:.2f}, fx={self.fx_rect:.2f}, B={self.baseline_rect:.4f}'
        )

        # Optional: quick sanity print (assume target ~7 m to infer baseline)
        B_implied = 7.0 * disp / self.fx_rect
        self.get_logger().info(f'Implied baseline if Z=7m: {B_implied:.4f} m')

        if 0.05 < depth < 100 and math.isfinite(depth):
            return depth
        self.get_logger().warn(f'Invalid depth {depth:.2f} from disparity={disp:.2f}')
        return 7.08

    # ===================== Callback ===================== #
    def _stereo_cb(self, left_msg, right_msg):
        try:
            self.left_rgb = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
            self.right_rgb = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
            self.get_logger().info(
                f'Received images: left={self.left_rgb.shape}, right={self.right_rgb.shape}'
            )

            disp = self._compute_disparity(self.left_rgb, self.right_rgb)
            if disp is None:
                return

            found, (cx, cy), mask_eroded, contour = self._detect_on_rectified()
            distance = self._depth_from_mask(mask_eroded)

            # Visualization overlay
            dv = np.where(disp < 0, 0, disp)
            dv = cv2.normalize(dv, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            dv = cv2.applyColorMap(dv, cv2.COLORMAP_JET)
            if found and contour is not None:
                cv2.drawContours(dv, [contour], -1, (255, 255, 255), 2)
                cv2.circle(dv, (int(cx), int(cy)), 6, (255, 255, 255), -1)
            cv2.imshow('Disparity Map (rectified)', dv)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f'Stereo callback error: {e}')

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DisparityDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down disparity debug node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()