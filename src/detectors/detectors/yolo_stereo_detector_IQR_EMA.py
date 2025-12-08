#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge
from ultralytics import YOLO
import time
import cv2
import numpy as np
import tempfile
import os
import shutil



class YoloStereoDetector(Node):
    def __init__(self):
        super().__init__('yolo_stereo_detector', namespace='follower_robot')

        self.bridge = CvBridge()

        model_name = 'yolo11n.pt'
        self.get_logger().info(f'Loading YOLO model: {model_name}')
        self.model = YOLO(model_name)
        self.class_names = self.model.model.names

        # ✅ NEW: only keep these classes
        self.allowed_classes = {"person", "fire hydrant", "car"}

        # topics you have
        left_topic = '/follower_robot/depth_cam/left/image_rect_color'
        right_topic = '/follower_robot/depth_cam/right/image_rect_color'

        self.create_subscription(Image, left_topic, self.left_cb, 10)
        self.create_subscription(Image, right_topic, self.right_cb, 10)

        self.left_pub = self.create_publisher(Detection2DArray, '/follower_robot/depth_cam/left/detections', 10)
        self.right_pub = self.create_publisher(Detection2DArray, '/follower_robot/depth_cam/right/detections', 10)

        self.get_logger().info('YOLO stereo detector is ready')
        self.last_run_time = 0.0
        self.yolo_period = 0.1
        self.tracker_config = "botsort.yaml"  # Or "bytetrack.yaml" for alternative

    def left_cb(self, msg: Image):
        self._run_yolo_and_publish(msg, 'left')

    def right_cb(self, msg: Image):
        self._run_yolo_and_publish(msg, 'right')

    def _run_yolo_and_publish(self, img_msg: Image, side: str):
        # 1) ROS Image -> OpenCV BGR8
        try:
            # Force a concrete encoding that YOLO & OpenCV like
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'[cv_bridge] failed on {side}: {e}')
            return

        self.get_logger().info(
            f'[{side}] after cv_bridge: type={type(cv_img)}, '
            f'dtype={getattr(cv_img, "dtype", None)}, '
            f'shape={getattr(cv_img, "shape", None)}'
        )

        # 2) Ensure we have a proper numpy uint8 HxWx3 array
        if not isinstance(cv_img, np.ndarray):
            try:
                cv_img = np.asarray(cv_img)
                self.get_logger().info(
                    f'[{side}] converted via np.asarray -> '
                    f'type={type(cv_img)}, dtype={cv_img.dtype}, shape={cv_img.shape}'
                )
            except Exception as e:
                self.get_logger().error(
                    f'[{side}] np.asarray(cv_img) failed: {e}'
                )
                return

        if cv_img is None:
            self.get_logger().error(f'[{side}] cv_img is None after conversion')
            return

        if cv_img.ndim != 3 or cv_img.shape[2] != 3:
            self.get_logger().error(
                f'[{side}] Unexpected image shape for YOLO: '
                f'{cv_img.shape}, dtype={cv_img.dtype}'
            )
            return

        if cv_img.dtype != np.uint8:
            self.get_logger().warn(
                f'[{side}] converting image dtype {cv_img.dtype} -> uint8 for YOLO'
            )
            cv_img = cv_img.astype(np.uint8)

        cv_bgr = np.ascontiguousarray(cv_img)

        # 3) DEBUG: local OpenCV test (copyMakeBorder) to be sure OpenCV is happy
        try:
            _border_test = cv2.copyMakeBorder(
                cv_bgr, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        except Exception as e:
            self.get_logger().error(
                f'[{side}] local copyMakeBorder test failed BEFORE YOLO: {e}'
            )
            return

        # 4) Save to a temporary file and let YOLO load from disk
        tmp_dir = tempfile.mkdtemp(prefix=f"yolo_{side}_")
        img_path = os.path.join(tmp_dir, "frame.png")

        try:
            ok = cv2.imwrite(img_path, cv_bgr)
            if not ok:
                self.get_logger().error(
                    f'[{side}] cv2.imwrite failed to write {img_path}'
                )
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return
        except Exception as e:
            self.get_logger().error(
                f'[{side}] cv2.imwrite error: {e}'
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        # 5) Run YOLO on the image path
        try:
            t0 = time.time()
            results = self.model.track(img_path, persist=True, tracker=self.tracker_config, imgsz=640, verbose=False)
            dt = time.time() - t0
            self.get_logger().info(f'YOLO {side} inference: {dt:.3f}s')
        except Exception as e:
            self.get_logger().error(f'[YOLO {side}] model.predict error: {e}')
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return
        finally:
            # Clean up temporary file/folder
            shutil.rmtree(tmp_dir, ignore_errors=True)

        out_msg = Detection2DArray()
        out_msg.header = img_msg.header

        if not results:
            # no detections
            if side == 'left':
                self.left_pub.publish(out_msg)
            else:
                self.right_pub.publish(out_msg)
            return

        # 6) Parse YOLO results
        vis = cv_bgr.copy()
        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.class_names.get(cls_id, str(cls_id))
                track_id = None
                if box.id is not None:
                    track_id = int(box.id.cpu().numpy())

                # filter: only person & fire hydrant
                if cls_name not in self.allowed_classes:
                    continue

                det = Detection2D()
                det.header = img_msg.header

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1

                det.bbox = BoundingBox2D()
                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
                det.bbox.size_x = float(w)
                det.bbox.size_y = float(h)

                hyp = ObjectHypothesisWithPose()
                det.id = str(track_id) if track_id is not None else ""  # Set Detection2D.id
                hyp.hypothesis.class_id = cls_name
                hyp.hypothesis.score = conf
                det.results.append(hyp)

                out_msg.detections.append(det)

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(
                    vis, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA
                )

        if side == 'left':
            self.left_pub.publish(out_msg)
        else:
            self.right_pub.publish(out_msg)

        # Optional visualization:
        #cv2.imshow(f'YOLO {side}', vis)
        #cv2.waitKey(1)





def main():
    rclpy.init()
    node = YoloStereoDetector()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
