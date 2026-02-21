#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

namespace gazebo
{

class MovingObstaclePlugin : public ModelPlugin
{
public:
  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
  {
    model = _model;
    world = model->GetWorld();

    // ---------------- Defaults (can be overridden from SDF) ----------------
    zone_radius = 8.0;
    wall_margin = 0.7;

    v0   = 1.2;
    v_max = 3.0;
    tau  = 0.6;

    // Social-force (repulsion) parameters
    A_robot = 8.0;  B_robot = 0.7;  R_robot = 2.0;
    A_obs   = 6.0;  B_obs   = 0.6;  R_obs   = 1.2;

    // Add GOAL repulsion (NEW)
    goal_name = "goal_marker";
    A_goal = 8.0;
    B_goal = 0.7;
    R_goal = 2.0;

    A_wall = 10.0;
    B_wall = 0.5;

    tangential_gain = 0.6;

    // Wander
    wander_heading_noise = 0.35;
    heading_change_rate  = 0.9;
    desired_heading = uniform(-M_PI, M_PI);

    // Names
    robot_name = "my_robot";
    obstacle_prefixes = {"obstacle_", "yolo_obstacle_"};

    // Stability
    max_acc = 6.0;
    vel_z_kill = true;

    // HARD collision-prevention radii (NEW)
    hard_robot_radius = 1.0;   // if closer than this -> emergency push
    hard_goal_radius  = 1.0;
    hard_obs_radius   = 0.8;

    // Wall turn behavior (NEW)
    wall_turn_cooldown_s = 0.8;        // don’t re-roll direction every tick
    last_wall_turn_time = 0.0;
    wall_turn_active_only_if_outward = true;

    // ---------------- Read SDF overrides ----------------
    if (_sdf)
    {
      if (_sdf->HasElement("zone_radius")) zone_radius = _sdf->Get<double>("zone_radius");
      if (_sdf->HasElement("wall_margin")) wall_margin = _sdf->Get<double>("wall_margin");

      if (_sdf->HasElement("v0")) v0 = _sdf->Get<double>("v0");
      if (_sdf->HasElement("v_max")) v_max = _sdf->Get<double>("v_max");
      if (_sdf->HasElement("tau")) tau = _sdf->Get<double>("tau");

      if (_sdf->HasElement("A_robot")) A_robot = _sdf->Get<double>("A_robot");
      if (_sdf->HasElement("B_robot")) B_robot = _sdf->Get<double>("B_robot");
      if (_sdf->HasElement("R_robot")) R_robot = _sdf->Get<double>("R_robot");

      if (_sdf->HasElement("A_obs")) A_obs = _sdf->Get<double>("A_obs");
      if (_sdf->HasElement("B_obs")) B_obs = _sdf->Get<double>("B_obs");
      if (_sdf->HasElement("R_obs")) R_obs = _sdf->Get<double>("R_obs");

      if (_sdf->HasElement("goal_name")) goal_name = _sdf->Get<std::string>("goal_name");
      if (_sdf->HasElement("A_goal")) A_goal = _sdf->Get<double>("A_goal");
      if (_sdf->HasElement("B_goal")) B_goal = _sdf->Get<double>("B_goal");
      if (_sdf->HasElement("R_goal")) R_goal = _sdf->Get<double>("R_goal");

      if (_sdf->HasElement("A_wall")) A_wall = _sdf->Get<double>("A_wall");
      if (_sdf->HasElement("B_wall")) B_wall = _sdf->Get<double>("B_wall");

      if (_sdf->HasElement("tangential_gain")) tangential_gain = _sdf->Get<double>("tangential_gain");

      if (_sdf->HasElement("wander_heading_noise")) wander_heading_noise = _sdf->Get<double>("wander_heading_noise");
      if (_sdf->HasElement("heading_change_rate")) heading_change_rate = _sdf->Get<double>("heading_change_rate");

      if (_sdf->HasElement("robot_name")) robot_name = _sdf->Get<std::string>("robot_name");

      if (_sdf->HasElement("hard_robot_radius")) hard_robot_radius = _sdf->Get<double>("hard_robot_radius");
      if (_sdf->HasElement("hard_goal_radius"))  hard_goal_radius  = _sdf->Get<double>("hard_goal_radius");
      if (_sdf->HasElement("hard_obs_radius"))   hard_obs_radius   = _sdf->Get<double>("hard_obs_radius");

      if (_sdf->HasElement("wall_turn_cooldown_s")) wall_turn_cooldown_s = _sdf->Get<double>("wall_turn_cooldown_s");
      if (_sdf->HasElement("wall_turn_active_only_if_outward"))
        wall_turn_active_only_if_outward = _sdf->Get<bool>("wall_turn_active_only_if_outward");
    }

    last_vel = ignition::math::Vector3d::Zero;
    last_update_time = world->SimTime();

    updateConnection = event::Events::ConnectWorldUpdateBegin(
      std::bind(&MovingObstaclePlugin::OnUpdate, this)
    );
  }

private:
  void OnUpdate()
  {
    const common::Time now = world->SimTime();
    double dt = (now - last_update_time).Double();
    if (dt <= 0.0) return;
    last_update_time = now;

    ignition::math::Vector3d pos = model->WorldPose().Pos();
    pos.Z() = 0.0;

    // Current velocity
    ignition::math::Vector3d v = model->WorldLinearVel();
    v.Z() = 0.0;

    // 1) Wander heading update (random walk)
    desired_heading += normal(0.0, wander_heading_noise) * dt;
    desired_heading = wrapPi(desired_heading);

    double drift = normal(0.0, 1.0) * heading_change_rate * dt * 0.2;
    desired_heading = wrapPi(desired_heading + drift);

    ignition::math::Vector3d e_des(std::cos(desired_heading), std::sin(desired_heading), 0.0);
    ignition::math::Vector3d v_des = v0 * e_des;

    // 2) Social forces
    ignition::math::Vector3d F = ignition::math::Vector3d::Zero;

    // 2a) Repulsion from robot
    physics::ModelPtr robot = world->ModelByName(robot_name);
    if (robot)
    {
      ignition::math::Vector3d rpos = robot->WorldPose().Pos();
      rpos.Z() = 0.0;
      F += repulsiveForce(pos, rpos, A_robot, B_robot, R_robot, tangential_gain);
    }

    // 2b) Repulsion from goal (NEW)
    physics::ModelPtr goal = world->ModelByName(goal_name);
    if (goal)
    {
      ignition::math::Vector3d gpos = goal->WorldPose().Pos();
      gpos.Z() = 0.0;
      F += repulsiveForce(pos, gpos, A_goal, B_goal, R_goal, tangential_gain);
    }

    // 2c) Repulsion from other obstacles
    for (auto m : world->Models())
    {
      if (!m) continue;
      if (m->GetName() == model->GetName()) continue;
      if (!isObstacleName(m->GetName())) continue;

      ignition::math::Vector3d opos = m->WorldPose().Pos();
      opos.Z() = 0.0;
      F += repulsiveForce(pos, opos, A_obs, B_obs, R_obs, tangential_gain);
    }

    // 2d) Boundary handling: inward force + RANDOM "bounce-turn" 120..240 deg (NEW)
    ignition::math::Vector3d center(0, 0, 0);
    ignition::math::Vector3d dc = pos - center;
    dc.Z() = 0.0;

    double dist = dc.Length();
    double edge_dist = zone_radius - dist; // positive inside, negative outside

    if (edge_dist < wall_margin)
    {
      ignition::math::Vector3d n_out = (dist > 1e-6) ? dc.Normalized() : ignition::math::Vector3d(1,0,0);
      ignition::math::Vector3d n_in  = -n_out;

      // Inward repulsion (keep yours)
      double d = std::max(1e-3, edge_dist);
      double mag = A_wall * std::exp((wall_margin - d) / std::max(1e-3, B_wall));
      F += mag * n_in;

      // Determine if we're moving outward
      bool moving_outward = (v.Dot(n_out) > 0.05); // >0 means velocity has outward component
      double tnow = now.Double();

      bool allowed = (tnow - last_wall_turn_time) > wall_turn_cooldown_s;
      bool should_turn = allowed && (!wall_turn_active_only_if_outward || moving_outward);

      if (should_turn)
      {
        // base direction: last velocity heading if moving, else current desired heading
        double base_heading = desired_heading;
        if (v.Length() > 0.05)
          base_heading = std::atan2(v.Y(), v.X());

        // Pick offset in [120°, 240°] = [2π/3, 4π/3] (opposite-ish)
        double off = uniform(2.0*M_PI/3.0, 4.0*M_PI/3.0);
        desired_heading = wrapPi(base_heading + off);

        last_wall_turn_time = tnow;
      }
    }

    // 3) Relaxation toward desired velocity + social forces
    ignition::math::Vector3d a = (v_des - v) / std::max(1e-3, tau) + F;

    // Clamp acceleration
    double a_len = a.Length();
    if (a_len > max_acc) a = a.Normalized() * max_acc;

    ignition::math::Vector3d v_cmd = v + a * dt;

    // Clamp speed
    double sp = v_cmd.Length();
    if (sp > v_max) v_cmd = v_cmd.Normalized() * v_max;

    if (vel_z_kill) v_cmd.Z() = 0.0;

    // 4) HARD safety separation (NEW): if too close, override velocity outward
    // Robot hard
    if (robot)
    {
      ignition::math::Vector3d rpos = robot->WorldPose().Pos();
      rpos.Z() = 0.0;
      double d = (pos - rpos).Length();
      if (d < hard_robot_radius && d > 1e-6)
      {
        ignition::math::Vector3d n = (pos - rpos) / d;
        v_cmd = std::min(v_max, v0) * n;  // push directly away
      }
    }
    // Goal hard
    if (goal)
    {
      ignition::math::Vector3d gpos = goal->WorldPose().Pos();
      gpos.Z() = 0.0;
      double d = (pos - gpos).Length();
      if (d < hard_goal_radius && d > 1e-6)
      {
        ignition::math::Vector3d n = (pos - gpos) / d;
        v_cmd = std::min(v_max, v0) * n;
      }
    }
    // Obstacles hard
    for (auto m : world->Models())
    {
      if (!m) continue;
      if (m->GetName() == model->GetName()) continue;
      if (!isObstacleName(m->GetName())) continue;

      ignition::math::Vector3d opos = m->WorldPose().Pos();
      opos.Z() = 0.0;
      double d = (pos - opos).Length();
      if (d < hard_obs_radius && d > 1e-6)
      {
        ignition::math::Vector3d n = (pos - opos) / d;
        v_cmd = std::min(v_max, v0) * n;
        break;
      }
    }

    // Apply
    model->SetLinearVel(v_cmd);
    model->SetAngularVel(ignition::math::Vector3d::Zero);

    last_vel = v_cmd;
  }

  // ---------------- Helpers ----------------
  ignition::math::Vector3d repulsiveForce(
      const ignition::math::Vector3d &self,
      const ignition::math::Vector3d &other,
      double A, double B, double R,
      double tangentialGain)
  {
    ignition::math::Vector3d d = self - other;
    d.Z() = 0.0;
    double dist = d.Length();
    if (dist < 1e-6) return ignition::math::Vector3d::Zero;

    ignition::math::Vector3d n = d / dist;

    double mag = A * std::exp((R - dist) / std::max(1e-3, B));
    ignition::math::Vector3d F = mag * n;

    ignition::math::Vector3d t(-n.Y(), n.X(), 0.0);
    double side = (self.X()*other.Y() - self.Y()*other.X()) >= 0.0 ? 1.0 : -1.0;
    F += tangentialGain * mag * side * t;

    return F;
  }

  bool isObstacleName(const std::string &name) const
  {
    for (const auto &p : obstacle_prefixes)
      if (name.find(p) != std::string::npos) return true;
    return false;
  }

  double wrapPi(double a) const
  {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
  }

  double uniform(double a, double b)
  {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(gen);
  }

  double normal(double mean, double stddev)
  {
    std::normal_distribution<double> dist(mean, stddev);
    return dist(gen);
  }

private:
  physics::ModelPtr model;
  physics::WorldPtr world;
  event::ConnectionPtr updateConnection;

  common::Time last_update_time;
  ignition::math::Vector3d last_vel;

  // Zone
  double zone_radius;
  double wall_margin;

  // Desired motion
  double v0;
  double v_max;
  double tau;

  // Social forces
  double A_robot, B_robot, R_robot;
  double A_obs,   B_obs,   R_obs;

  // GOAL avoidance (NEW)
  std::string goal_name;
  double A_goal, B_goal, R_goal;

  double A_wall,  B_wall;
  double tangential_gain;

  // Wander
  double wander_heading_noise;
  double heading_change_rate;
  double desired_heading;

  // Stability
  double max_acc;
  bool vel_z_kill;

  // Names
  std::string robot_name;
  std::vector<std::string> obstacle_prefixes;

  // Hard separation (NEW)
  double hard_robot_radius;
  double hard_goal_radius;
  double hard_obs_radius;

  // Wall bounce behavior (NEW)
  double wall_turn_cooldown_s;
  double last_wall_turn_time;
  bool wall_turn_active_only_if_outward;

  // RNG
  std::default_random_engine gen;
};

GZ_REGISTER_MODEL_PLUGIN(MovingObstaclePlugin)

} // namespace gazebo
