#include <cmath>
#include <string>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/float32.hpp"

using std::placeholders::_1;

static inline bool is_bad(float x) {
  return !std::isfinite(x);
}

class UltrasonicRangeBridge : public rclcpp::Node {
public:
  UltrasonicRangeBridge()
  : Node("ultrasonic_range_bridge")
  {
    // Input topics (from your SRF05 plugins)
    const std::unordered_map<std::string, std::string> inputs = {
      {"front_left",  "/follower_robot/srf05_front_left_plugin/out"},
      {"front_right", "/follower_robot/srf05_front_right_plugin/out"},
      {"left_side",   "/follower_robot/srf05_left_side_plugin/out"},
      {"right_side",  "/follower_robot/srf05_right_side_plugin/out"},
    };

    // Output topic prefix
    const std::string out_prefix = "/follower_robot/ultrasonic_bridge/";

    for (const auto &kv : inputs) {
      const auto &name = kv.first;
      const auto &topic = kv.second;

      // Publisher: clean Range
      pubs_range_[name] = this->create_publisher<sensor_msgs::msg::Range>(
        out_prefix + name + "/range", rclcpp::QoS(10).reliable()
      );

      // Publisher: clean float distance
      pubs_dist_[name] = this->create_publisher<std_msgs::msg::Float32>(
        out_prefix + name + "/distance_m", rclcpp::QoS(10).reliable()
      );

      // Subscriber
      subs_[name] = this->create_subscription<sensor_msgs::msg::Range>(
        topic,
        rclcpp::QoS(10).reliable(),
        [this, name](sensor_msgs::msg::Range::SharedPtr msg) {
          this->on_range(name, *msg);
        }
      );

      RCLCPP_INFO(this->get_logger(), "Bridging %s  ->  %s%s",
                  topic.c_str(), out_prefix.c_str(), name.c_str());
    }

    RCLCPP_INFO(this->get_logger(), "UltrasonicRangeBridge READY");
  }

private:
  void on_range(const std::string &name, const sensor_msgs::msg::Range &in)
  {
    // Make a copy and sanitize fields
    sensor_msgs::msg::Range out = in;

    // Fix min/max if invalid
    if (is_bad(out.min_range) || out.min_range < 0.0f) out.min_range = 0.02f;
    if (is_bad(out.max_range) || out.max_range <= out.min_range) out.max_range = 5.0f;

    // Fix range if invalid
    float r = out.range;
    if (is_bad(r) || r < 0.0f) {
      r = out.max_range;  // treat as "no obstacle"
    }

    // Clamp
    if (r < out.min_range) r = out.min_range;
    if (r > out.max_range) r = out.max_range;

    out.range = r;

    // Publish clean Range
    pubs_range_[name]->publish(out);

    // Publish clean float
    std_msgs::msg::Float32 d;
    d.data = r;
    pubs_dist_[name]->publish(d);
  }

  std::unordered_map<std::string, rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr> subs_;
  std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr> pubs_range_;
  std::unordered_map<std::string, rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr> pubs_dist_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<UltrasonicRangeBridge>());
  rclcpp::shutdown();
  return 0;
}
