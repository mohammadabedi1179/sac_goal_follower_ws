#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import json
import sys
from pathlib import Path
import importlib.util

# ---------- Locate repo root & visualDet3D ----------
THIS_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(10):
        # we are looking for .../src/external/visualDet3D
        if (cur / "src" / "external" / "visualDet3D").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start  # fallback


REPO_ROOT = find_repo_root(THIS_DIR)
VISUALDET_ROOT = REPO_ROOT / "src" / "external" / "visualDet3D"

# Make both the repo root and the visualDet3D package importable
for p in (VISUALDET_ROOT, VISUALDET_ROOT / "visualDet3D"):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)

# --------- Load cfg directly from config/my_robot_stereo/config.py ----------
CFG_PATH = VISUALDET_ROOT / "config" / "my_robot_stereo" / "config.py"


def load_cfg(path: Path):
    """Dynamically load the config module and return its `cfg` object."""
    spec = importlib.util.spec_from_file_location(
        "my_robot_stereo_cfg", str(path)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # config.py defines `cfg = edict(...)`
    return module.cfg


from visualDet3D.networks.detectors.yolostereo3d_detector import Stereo3D
from visualDet3D.data.pipeline import build_augmentator


class Stereo3DLiveNode(Node):
    def __init__(self):
        super().__init__("stereo3d_live", namespace="follower_robot")
        self.bridge = CvBridge()

        # 🔹 Load full cfg (with cfg.detector, cfg.path, cfg.data, ...)
        self.cfg = load_cfg(CFG_PATH)

        # network_cfg for Stereo3D is cfg.detector
        detector_cfg = getattr(self.cfg, "detector", self.cfg)

        # Build model
        self.model = Stereo3D(detector_cfg)

        # Load checkpoint from cfg.path.checkpoint_path
        ckpt_path = self.cfg.path.checkpoint_path
        self.get_logger().info(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            self.get_logger().info("Model moved to GPU")

        # 🔹 Build test-time augmentation (same as KittiStereoTestDataset)
        if hasattr(self.cfg, "data") and hasattr(self.cfg.data, "test_augmentation"):
            self.transform = build_augmentator(self.cfg.data.test_augmentation)
            self.get_logger().info("Using cfg.data.test_augmentation pipeline")
        else:
            self.transform = None
            self.get_logger().warn(
                "No cfg.data.test_augmentation found – images will be only normalized to [0,1]"
            )

        # 🔹 Build projection matrices P2, P3 from cfg.calib
        self.P2_base, self.P3_base = self._build_projection_mats()
        self.get_logger().info(f"P2_base:\n{self.P2_base}")
        self.get_logger().info(f"P3_base:\n{self.P3_base}")

        # Subscriptions
        self.left_sub = self.create_subscription(
            Image,
            "/follower_robot/depth_cam/left/image_rect_color",
            self.left_cb,
            10,
        )
        self.right_sub = self.create_subscription(
            Image,
            "/follower_robot/depth_cam/right/image_rect_color",
            self.right_cb,
            10,
        )

        self.pub = self.create_publisher(
            String, "/follower_robot/obstacles_depth", 10
        )

        self.left_img = None
        self.right_img = None
        self.frame_counter = 0

        self.get_logger().info(
            "YOLOStereo3D LIVE node ready – waiting for synced images"
        )

    # ---------------- Camera / projection helpers ----------------

    def _build_projection_mats(self):
        """
        Build simple pinhole P2/P3 matrices from cfg.calib.{focal_length, baseline, cx, cy}.
        Shape: [3,4] each.
        """
        if not hasattr(self.cfg, "calib"):
            # Fallback generic camera
            fx = fy = 700.0
            cx, cy, b = 320.0, 240.0, 0.1
        else:
            fx = float(self.cfg.calib.get("focal_length", 700.0))
            fy = fx
            cx = float(self.cfg.calib.get("cx", 320.0))
            cy = float(self.cfg.calib.get("cy", 240.0))
            b = float(self.cfg.calib.get("baseline", 0.1))

        P2 = np.array(
            [
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        # Right camera is translated along X by baseline
        P3 = np.array(
            [
                [fx, 0.0, cx, -fx * b],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        return P2, P3

    # ---------------- ROS callbacks ----------------

    def left_cb(self, msg: Image):
        self.left_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.try_infer()

    def right_cb(self, msg: Image):
        self.right_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.try_infer()

    def try_infer(self):
        # Need both images
        if self.left_img is None or self.right_img is None:
            return

        # Throttle: run every 3 frames
        self.frame_counter += 1
        if self.frame_counter % 3 != 0:
            return

        results = self.infer_once(self.left_img, self.right_img)

        if results:
            msg_out = String()
            msg_out.data = json.dumps(results)
            self.pub.publish(msg_out)
            self.get_logger().info(f"Depths → {results}")
        else:
            self.get_logger().info("Stereo3D: no objects detected in this frame")

    # ---------------- Core inference ----------------

    def infer_once(self, left_bgr, right_bgr):
        """
        Run Stereo3D.test_forward on a single stereo pair.
        """
        # 1) BGR -> RGB (model was trained on RGB KITTI images)
        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

        # 2) Apply test-time augment (same as KittiStereoTestDataset) if available
        if self.transform is not None:
            left_proc, right_proc, P2, P3 = self.transform(
                left_rgb,
                right_rgb,
                self.P2_base.copy(),
                self.P3_base.copy(),
            )
        else:
            left_proc = left_rgb.astype(np.float32) / 255.0
            right_proc = right_rgb.astype(np.float32) / 255.0
            P2 = self.P2_base.copy()
            P3 = self.P3_base.copy()

        # 3) Numpy HWC -> Torch BCHW
        left_tensor = torch.from_numpy(left_proc).permute(2, 0, 1).unsqueeze(0).float()
        right_tensor = (
            torch.from_numpy(right_proc).permute(2, 0, 1).unsqueeze(0).float()
        )

        # 4) Projection matrices as [B, 3, 4]
        P2_tensor = torch.from_numpy(P2).unsqueeze(0).float()
        P3_tensor = torch.from_numpy(P3).unsqueeze(0).float()

        # Move to same device as model
        device = next(self.model.parameters()).device
        left_tensor = left_tensor.to(device)
        right_tensor = right_tensor.to(device)
        P2_tensor = P2_tensor.to(device)
        P3_tensor = P3_tensor.to(device)

        # 5) Forward pass
        with torch.no_grad():
            scores, bbox3d, cls_idxs = self.model(
                [left_tensor, right_tensor, P2_tensor, P3_tensor]
            )

        self.get_logger().info(
            f"Stereo3D raw: scores type={type(scores)}, "
            f"bbox3d type={type(bbox3d)}, cls_idxs type={type(cls_idxs)}"
        )

        # ---- unwrap list outputs [scores], [bboxes], [labels] ----
        if isinstance(scores, (list, tuple)):
            if len(scores) == 0:
                self.get_logger().warn("Stereo3D: empty score list from model.")
                return []
            scores = scores[0]
            bbox3d = bbox3d[0]
            cls_idxs = cls_idxs[0]

        # Debug a few boxes before conversion
        try:
            n_boxes = scores.shape[0]
            self.get_logger().info(
                f"Stereo3D: model returned {int(n_boxes)} boxes BEFORE filtering"
            )
            for i in range(min(5, n_boxes)):
                self.get_logger().info(
                    f"  box {i}: score={float(scores[i]):.4f}, "
                    f"depth_z={float(bbox3d[i, 2]):.3f}, cls_id={int(cls_idxs[i])}"
                )
        except Exception as e:
            self.get_logger().warn(f"Stereo3D: debug-shape error: {e}")

        # ---- convert to numpy ----
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            if isinstance(x, np.ndarray):
                return x
            raise TypeError(f"Unexpected type in to_numpy: {type(x)}")

        try:
            scores = to_numpy(scores)      # [N]
            bbox3d = to_numpy(bbox3d)      # [N, 7] (x,y,z,w,h,l,alpha)
            cls_idxs = to_numpy(cls_idxs)  # [N]
        except Exception as e:
            self.get_logger().error(f"Stereo3D: to_numpy failed: {e}")
            return []

        # safety: if nothing detected
        if scores.size == 0 or bbox3d.size == 0:
            self.get_logger().info("Stereo3D: no boxes from model after conversion")
            return []

        # ----------------- Build raw result list -----------------
        results = []

        # base detection threshold from cfg
        test_cfg = getattr(self.cfg, "test_cfg", None)
        score_thr = getattr(test_cfg, "score_thr", 0.35) if test_cfg is not None else 0.35
        if not isinstance(score_thr, (float, int)):
            score_thr = 0.35

        # helper to map class id -> name
        def map_class_id(c_id: int) -> str:
            if hasattr(self.cfg, "class_names") and 0 <= c_id < len(self.cfg.class_names):
                name = self.cfg.class_names[c_id]
            elif hasattr(self.cfg, "obj_types") and 0 <= c_id < len(self.cfg.obj_types):
                name = self.cfg.obj_types[c_id]
            else:
                name = f"class_{c_id}"

            # simplify some labels if you like
            if name.lower().startswith("pedestrian"):
                return "person"
            if name.lower().startswith("car"):
                return "car"
            return name

        N = scores.shape[0]
        for i in range(N):
            s = float(scores[i])
            if s < score_thr:
                continue

            state = bbox3d[i]
            if state.shape[0] < 3:
                continue

            depth = float(state[2])   # z in camera coordinates

            # filter crazy depths (negative / too close / too far)
            if not (0.3 <= depth <= 80.0):
                continue

            cls_id = int(cls_idxs[i])
            cls_name = map_class_id(cls_id)

            results.append(
                {
                    "class": cls_name,
                    "depth_m": depth,
                    "score": s,
                }
            )

        if not results:
            self.get_logger().info("Stereo3D: no valid 3D boxes after filtering")
            return []

        # ----------------- Compress results -----------------
        # 1) Keep top-K by score so we don't spam anchors
        TOP_K = 50
        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:TOP_K]

        # 2) For each class, keep the **closest** one (if depths tie, higher score)
        best_by_class = {}
        for r in results:
            c = r["class"]
            if c not in best_by_class:
                best_by_class[c] = r
            else:
                old = best_by_class[c]
                if (r["depth_m"] < old["depth_m"] - 1e-3 or
                        (abs(r["depth_m"] - old["depth_m"]) < 1e-3 and r["score"] > old["score"])):
                    best_by_class[c] = r

        final_results = []
        for c, r in best_by_class.items():
            final_results.append(
                {
                    "class": c,
                    "depth_m": round(r["depth_m"], 3),
                    "score": round(r["score"], 3),
                }
            )

        self.get_logger().info(
            f"Stereo3D: {len(final_results)} objects → {final_results}"
        )

        return final_results



def main():
    rclpy.init()
    node = Stereo3DLiveNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
