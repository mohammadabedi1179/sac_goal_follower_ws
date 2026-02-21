#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo
{
  class DynamicObstaclePlugin : public ModelPlugin
  {
  public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;
      this->world = _model->GetWorld();

      // Read parameters from SDF
      this->speed = _sdf->Get<double>("speed", 0.5).first;
      this->boundary = _sdf->Get<double>("boundary", 8.0).first;

      // Random initial direction
      double theta = ((double) rand() / RAND_MAX) * 2.0 * M_PI;
      this->vx = speed * cos(theta);
      this->vy = speed * sin(theta);

      this->updateConnection =
        event::Events::ConnectWorldUpdateBegin(
          std::bind(&DynamicObstaclePlugin::OnUpdate, this));

      std::cout << "[DynamicObstaclePlugin] Loaded for "
                << this->model->GetName()
                << " speed=" << speed << std::endl;
    }

  public:
    void OnUpdate()
    {
      ignition::math::Pose3d pose = this->model->WorldPose();
      double x = pose.Pos().X();
      double y = pose.Pos().Y();

      // Boundary reflection
      if (abs(x) > boundary)
        vx = -vx;

      if (abs(y) > boundary)
        vy = -vy;

      ignition::math::Vector3d vel(vx, vy, 0.0);
      this->model->SetLinearVel(vel);
    }

  private:
    physics::ModelPtr model;
    physics::WorldPtr world;
    event::ConnectionPtr updateConnection;

    double vx;
    double vy;
    double speed;
    double boundary;
  };

  GZ_REGISTER_MODEL_PLUGIN(DynamicObstaclePlugin)
}
