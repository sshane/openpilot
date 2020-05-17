import json
import ast
import time

f = '{\n  "alca_min_speed": 25.0,\n  "alca_nudge_required": true,\n  "awareness_factor": 3.0,\n  "camera_offset": 0.06,\n  "disengage_on_gas": true,\n  "dynamic_follow": "auto",\n  "dynamic_gas": true,\n  "enable_long_derivative": false,\n  "hide_auto_df_alerts": false,\n  "lane_hug_angle_offset": 0.0,\n  "lane_hug_direction": null,\n  "log_data": false,\n  "min_dynamic_lane_speed": 20.0,\n  "no_ota_updates": false,\n  "op_edit_live_mode": false,\n  "steer_ratio": null,\n  "test_param": [\n    0,\n    5,\n    99.85,\n    45.45\n  ],\n  "test_param1": 45.987,\n  "upload_on_hotspot": false,\n  "use_dynamic_lane_speed": true,\n  "v_rel_acc_modifier": 1.0\n}'

t = time.time()
for _ in range(100000):
  j = json.loads(f)
print(time.time() - t)

f = f.replace('\n', '').replace(' ', '').replace('null', 'None').replace('false', 'False').replace('true', 'True')

t = time.time()
for _ in range(100000):
  a = ast.literal_eval(f)
print(time.time() - t)
