std_msgs/Header header

bool    visible        # true if red goal marker detected in current frame

float32 depth_m        # smoothed depth [m] (camera -> marker)
float32 bearing_rad    # bearing in camera frame [rad] (left camera)

float32 cx             # marker center x [px] in left image
float32 cy             # marker center y [px] in left image
