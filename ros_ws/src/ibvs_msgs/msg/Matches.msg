std_msgs/Header header

# one entry per match
uint32[] ref_id

# flattened: [x0,y0,x1,y1,...] for the *current* measurement positions
float32[] xy

# similarity per match (cosine)
float32[] sim
