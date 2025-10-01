#!/usr/bin/env python3
import math, random
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, Quaternion, Point, Vector3
from gazebo_msgs.srv import GetEntityState, SetEntityState
from gazebo_msgs.msg import EntityState

def yaw_from_quat(q: Quaternion):
    # q -> yaw
    s = 2.0*(q.w*q.z + q.x*q.y)
    c = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
    return math.atan2(s, c)

def quat_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw/2.0), w=math.cos(yaw/2.0))

def clamp(a, lo, hi): return lo if a < lo else hi if a > hi else a

class MovingGoal(Node):
    def __init__(self):
        super().__init__('moving_goal')

        # Params
        self.declare_parameter('entity_name', 'goal_marker')
        self.declare_parameter('get_state_srv', '/get_entity_state')
        self.declare_parameter('set_state_srv', '/set_entity_state')
        self.declare_parameter('center_x', 20.0)
        self.declare_parameter('center_y', 25.0)
        self.declare_parameter('radius',   5.0)
        self.declare_parameter('ground_z', 0.15)
        self.declare_parameter('v_lin',    0.6)   # m/s nominal forward speed
        self.declare_parameter('v_ang',    0.4)   # rad/s nominal turn rate
        self.declare_parameter('dt',       0.05)  # s, timer period
        self.declare_parameter('jitter_turn', 0.2) # max random extra turn rate

        self.name = self.get_parameter('entity_name').get_parameter_value().string_value
        self.get_srv_name = self.get_parameter('get_state_srv').get_parameter_value().string_value
        self.set_srv_name = self.get_parameter('set_state_srv').get_parameter_value().string_value
        self.cx = float(self.get_parameter('center_x').value)
        self.cy = float(self.get_parameter('center_y').value)
        self.R  = float(self.get_parameter('radius').value)
        self.ground_z = float(self.get_parameter('ground_z').value)
        self.v_lin = float(self.get_parameter('v_lin').value)
        self.v_ang = float(self.get_parameter('v_ang').value)
        self.dt = float(self.get_parameter('dt').value)
        self.jitter_turn = float(self.get_parameter('jitter_turn').value)

        self.get_cli = self.create_client(GetEntityState, self.get_srv_name)
        self.set_cli = self.create_client(SetEntityState, self.set_srv_name)

        # Wait for services
        self.get_logger().info(f"Waiting for {self.get_srv_name} and {self.set_srv_name} ...")
        while not self.get_cli.wait_for_service(timeout_sec=1.0): pass
        while not self.set_cli.wait_for_service(timeout_sec=1.0): pass

        # Seed pose/yaw from sim
        ok = self.try_read_pose()
        if not ok:
            self.x, self.y, self.yaw = self.cx, self.cy, 0.0
        self.get_logger().info(f"Start at ({self.x:.2f}, {self.y:.2f})")

        # Small random initial heading
        self.yaw += random.uniform(-math.pi, math.pi)

        self.timer = self.create_timer(self.dt, self.step)
        self.not_found_warned = False

    def try_read_pose(self) -> bool:
        req = GetEntityState.Request()
        req.name = self.name
        req.reference_frame = ''  # world
        try:
            resp = self.get_cli.call(req)
        except Exception as e:
            self.get_logger().warn(f"get_entity_state call failed: {e}")
            return False

        # Treat success=True as authoritative (some Gazebo builds leave state.name blank)
        if not resp.success:
            return False

        p = resp.state.pose.position
        q = resp.state.pose.orientation
        self.x, self.y = float(p.x), float(p.y)
        self.yaw = yaw_from_quat(q)
        return True

    def step(self):
        # Ensure entity exists (robust to empty state.name)
        if not self.try_read_pose():
            if not self.not_found_warned:
                self.get_logger().warn("Entity not found yet; will retry.")
                self.not_found_warned = True
            return
        self.not_found_warned = False

        # Random walk + boundary keeping inside circle
        # Vector from center
        dx = self.x - self.cx
        dy = self.y - self.cy
        r = math.hypot(dx, dy)

        # Boundary steering: if near/outside boundary, bias yaw back inward
        if r > self.R:
            desired = math.atan2(self.cy - self.y, self.cx - self.x)
            err = (desired - self.yaw + math.pi) % (2*math.pi) - math.pi
            w = clamp(self.v_ang * 2.0 * err, -1.2*self.v_ang, 1.2*self.v_ang)
        elif r > 0.8*self.R:
            desired = math.atan2(self.cy - self.y, self.cx - self.x)
            err = (desired - self.yaw + math.pi) % (2*math.pi) - math.pi
            w = clamp(self.v_ang * 1.2 * err, -self.v_ang, self.v_ang)
        else:
            # free roam with some jitter
            w = random.uniform(-self.jitter_turn, self.jitter_turn)

        v = self.v_lin

        # Integrate
        self.yaw += w * self.dt
        self.x   += v * math.cos(self.yaw) * self.dt
        self.y   += v * math.sin(self.yaw) * self.dt

        # Send small incremental pose update
        pose = Pose()
        pose.position = Point(x=self.x, y=self.y, z=self.ground_z)
        pose.orientation = quat_from_yaw(self.yaw)

        state = EntityState(
            name=self.name,
            pose=pose,
            twist=Twist(linear=Vector3(), angular=Vector3()),
            reference_frame='world'
        )

        req = SetEntityState.Request(state=state)
        try:
            _ = self.set_cli.call(req)
        except Exception as e:
            self.get_logger().warn(f"set_entity_state failed: {e}")

def main():
    rclpy.init()
    node = MovingGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
