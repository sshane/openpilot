**Update: The `stock_additions` branch will now be the main branch that will receive occasional and verified updates from my development branches. To ensure an up to date experience, checkout and drive on the main `stock_additions` branch! Run this command on your EON to set everything up:**

`cd /data/openpilot;git fetch;git pull;git checkout stock_additions;git reset --hard origin/stock_additions;git checkout stock_additions;git clean -xfd;git pull`

Then run `git branch` to make sure you are on `stock_additions`. The command above is just a one time process for users that have cloned the branch in the past and resolves conflicts.

Shane's Stock Additions 0.7.1 (version 0.2)
=====

This branch is simply stock openpilot with some additions to help it drive as smooth as possible on my 2017 Toyota Corolla w/ comma pedal.


Highlight Features
=====

* [**Dynamic follow (now with profiles!)**](#dynamic-follow-3-profiles)
* [**Dynamic gas**](#dynamic-gas)
* [**Customize this branch (opEdit Parameter class)**](#Customize-this-branch-opEdit-Parameter-class)
* [**Custom wheel offset to reduce lane hugging**](#Custom-wheel-offset-to-reduce-lane-hugging)
* [**Live tuning support**](#Live-tuning-support)

-----

Dynamic follow (3 profiles)
-----
This is my dynamic follow from 0.5, where it changes your TR (following distance) dynamically based on multiple vehicle factors, as well as data from the lead vehicle. [Here's an old write up from a while ago explaining how it works exactly. Some of it might be out of date, but how it functions is the same.](https://github.com/ShaneSmiskol/openpilot/blob/dynamic-follow/README.md) The goal is to essentially make the driving experience more smooth and increase safety, braking and accelerating sooner.

Now you can choose a profile based on traffic and your driving preference. There are three profiles currently:
  * `traffic` - Meant to keep you a bit closer in traffic, hopefully reducing cut-ins. May not be the safest when approaching a vastly slower vehicle.
  * `relaxed` - This is the current and now default dynamic follow profile just with a cool name. Also slight closer than previously at high speeds.
  * `roadtrip` - This profile is for road trips mainly where you're on two lane highways and don't want to be following particularly closely; at night for example.
<img src=".media/photos/df_profiles.png?raw=true" width="800">

**How to choose a profile:** The easiest way is to use `opEdit` over ssh with your mobile device.
```python
cd /data/openpilot
python op_edit.py
```
Then change the `dynamic_follow` parameter to one of the above profiles, make sure to spell it correctly. You're done!

Dynamic gas
-----
Currently supported vehicles (w/ comma pedal only):
  * 2017 Toyota Corolla (non-TSS2)
  * Toyota RAV4 (non-TSS2)
  * 2019 Honda Pilot

*TODO: Need to factor in distance, as it will not accelerate to get closer to the stopped lead if you engage at ~0mph far back from the lead.*

This aims to provide a smoother driving experience in stop and go traffic (under 20 mph) by modifying the maximum gas that can be applied based on your current velocity and the relative velocity of the lead car. It'll also of course increase the maximum gas when the lead is accelerating to help you get up to speed quicker than stock. And smoother; this eliminates the jerking you get from stock openpilot with comma pedal. It tries to coast if the lead is only moving slowly, it doesn't use maximum gas as soon as the lead inches forward :). When you are above 20 mph, relative velocity and the following distance is taken into consideration.

Customize this branch (opEdit Parameter class)
-----
This is a handy tool to change your `opParams` parameters without diving into any json files or code. You can specify parameters to be used in any fork's operation that supports `opParams`. First, ssh in to your EON and make sure you're in `/data/openpilot`, then start `opEdit`:
```python
cd /data/openpilot
python op_edit.py
```

Some parameters you can use to customize this fork:
- `camera_offset`: Your camera offset to use in lane_planner.py. Helps fix lane hugging
- `awareness_factor`: The multiplier for driver monitoring
- `alca_nudge_required`: Whether to wait for applied torque to the wheel (nudge) before making lane changes
- `alca_min_speed`: The minimum speed allowed for an automatic lane change
- `static_steer_ratio`: Whether you want to use the learned steer ratio or the static one in your interface.py
- `upload_on_hotspot`: Controls whether your EON will upload driving log data on your phone's hotspot
- `reset_integral`: Resets integral gain whenever the longitudinal PID error crosses or is zero. Helps overshoot

A list of parameters that you can modify are located [here](common/op_params.py#L42).

Parameters are stored at `/data/op_params.json`

Dynamic lane speed
-----
*This feature is disabled until I can figure out how to improve it.*

This is a feature that reduces your cruising speed if many vehicles around you are significantly slower than you. This works with and without an openpilot-identified lead. Ex.: It will slow you down if traveling in an open lane with cars in adjacent lanes that are slower than you. Or if the lead in front of the lead is slowing down, as well as cars in other lanes far ahead. The most it will slow you down is some average of: (the set speed and the average of the surrounding cars) The more the radar points, the more weight goes to the speeds of surrounding vehicles.

Custom wheel offset to reduce lane hugging
-----
***Update**: The performance of this modification is iffy at best, but it definitely is more apparant than just tuning your camera offset value. Removing the immediate angle offset can have some weird oscilating effect when it's windy or on roads with camber (slant to one side). Will be left in, but disabled.*

Stock openpilot doesn't seem to be able to identify your car's true angle offset. With the `LaneHugging` module you can specify a custom angle offset to be added to your desired steering angle. Simply find the angle your wheel is at when you're driving on a straight highway. By default, this is disabled, to enable you can:
- Use the `opEdit` class in the root directory of openpilot. To use it, simply open an `ssh` shell and enter the commands below:
    ```python
    cd /data/openpilot
    python op_edit.py
    ```
    You'll be greeted with a list of your parameters you can explore, enter the number corresponding to `lane_hug_direction`. Your options are to enter `'left'` or `'right'` for whichever direction your car has a tendency to hug toward. `None` will disable the feature.
    Finally you'll need to enter your absolute angle offset (negative will be converted to positive) with the `opParams` parameter: `lane_hug_angle_offset`.

~~Two PID loops to control gas and brakes independently~~
-----
***Update**: Probably going to remove this addition, as tuning the current pedal parameters will be a more robust solution in the long run.*

If you have a Toyota Corolla with a comma pedal, you'll love this addition. Two longitudinal PID loops are set up in `longcontrol.py` so that one is running with comma pedal tuning to control the gas, and the other is running stock non-pedal tuning for better braking control. In the car, this feels miles better than stock openpilot, and nearly as good as your stock Toyota cruise control before you pulled out your DSU! It won't accelerate up to stopped cars and brake at the last moment anymore.

Live tuning support
-----
Currently only the `camera_offset`, `lane_hug_angle_offset`, `dynamic_follow` parameters are supported.
- Just start opEdit with the instructions above and pick a parameter. It will let you know if it supports live tuning, if so, updates will take affect within 5 seconds!
- Alternatively, you can use the new opTune module to live tune quicker and easier! It stays in the parameter edit view so you can more easily experiment with values. opTune shown below:

<img src=".media/gifs/op_tune.gif?raw=true" width="600">
