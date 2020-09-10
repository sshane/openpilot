Stock Additions v0.1.1 (0.7.7)
===
 * Disable automatic updates by default and show changelog to users in apk
 * LaneSpeed process is much more efficient, CPU usage down from 6% (causing lag alerts) to ~1%
   * Use numpy_fast.interp instead of np.interp for slight boost in performance
 * Support white panda with experimental parameter. localizer performance may be reduced
 * Fixed silent alerts due to the new alert manager in 0.7.7
 * Make all the UI buttons show their state with unique colors!
   * Also added more colors to opEdit!
 * Add derivative to lateral PI control for TSS1 & TSS2 Corolla and Prius!
 * Add a Reset to origin button to TextWindow when an error occurs
 * Color coded model lane lines and path
 * Add ZSS support

Stock Additions v0.1 (0.7.7)
===
 * LaneSpeed process is much more efficient, CPU usage down from 6% (causing lag alerts) to ~1%
   * Use numpy_fast.interp instead of np.interp for slight boost in performance
 * Support white panda with experimental parameter. localizer performance may be reduced
 * Fixed silent alerts due to the new alert manager in 0.7.7
 * Make all the UI buttons show their state with unique colors!
   * Also added more colors to opEdit!
 * Add derivative to lateral PI control for TSS1 & TSS2 Corolla and Prius!
 * Add a Reset to origin button to TextWindow when an error occurs
 * Color coded model lane lines and path
 * Add ZSS support

Stock Additions 0.7.5
===
 * Various opEdit improvements, colors, highlighting of last explored param, quicker messages
 * New Dynamic Follow modification to help slow and accelerate sooner
 * Live parameters update quicker
 * dfManager reliability improvements
 * New LaneSpeed and Dynamic Camera Offset features added

Stock Additions 0.7.4 (version 0.2)
===
 * Sidebar will not pop out when you tap the Dynamic Follow profile change button when driving
 * Add derivative to longcontrol. Improves responsiveness and helps overshoot

Stock Additions 0.7.1
===
 * Recover faster when lead leaves path, or when braking or acceleration is required immediately. Also should speed up the acceleration with auto lane change.
 * Add 3 different dynamic follow profiles: `roadtrip`, `relaxed`, and `traffic`. `relaxed` is the current dynamic follow profile. You can also swap profiles without rebooting live by using opEdit. SSH in and: `cd /data/openpilot;python op_edit.py` The parameter to change is `dynamic_follow`
 * Fix for steering unavailable when wheel goes over 100 degrees/sec.
 * Tuning for dynamic gas
 * Accelerate quicker and get closer when you're lane changing
 * Higher acceleration, and higher limits for turns
 * Automatic updates, EON will reboot by itself as long as it is inactive

Stock Additions 0.7
===
 * Dynamic lane speed is a new feature that reduces your cruising speed if many vehicles around you are significantly slower than you. This works with and without an openpilot-identified lead.
 * Dynamic gas tuning. Above 20 mph we take lead velocity and the following distance into account. Possibility of different tuning for different cars in the future. (DYNAMIC GAS NOW ONLY WORKS ON TOYOTA COROLLA AND RAV4 PEDAL)
 * Dynamic follow tuning, don't get as close when lead is accelerating.
 * Added static_steer_ratio parameter, if True openpilot will use the steer ratio in your interface file. Default is true, false uses the openpilot learned value which can vary through your drives.
 * Added ability to live tune parameters with `op_tune.py`. Currently only the camera offset (`camera_offset`) is supported.
 * Some Corolla tuning.
 * Reduce max acceleration.
 * TO NOTE: Dynamic Lane Speed will not work with stopped cars, at any speed. There is also a margin that cars must be traveling within in order to affect your speed. Don't expect anything magical, just minor quality of drive improvements.
