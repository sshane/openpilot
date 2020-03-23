Shane's Stock Additions 0.7.4
=====

This branch is simply stock openpilot with some additions to help it drive as smooth as possible on my 2017 Toyota Corolla w/ comma pedal.

-----

Highlight Features
=====

* [**Dynamic follow (now with profiles!)**](#dynamic-follow-3-profiles)
* [**Dynamic gas**](#dynamic-gas)
* [**Customize this branch (opEdit Parameter class)**](#Customize-this-branch-opEdit-Parameter-class)
* [**Live tuning support**](#Live-tuning-support)
* [**Custom wheel offset to reduce lane hugging**](#Custom-wheel-offset-to-reduce-lane-hugging)
* [**Automatic updates**](#Automatic-updates)

Documentation
=====
* [**Quick Installation**](#Quick-installation)
* [**Branches**](#Branches)

-----

Dynamic follow (3 profiles)
-----
This is my dynamic follow from 0.5, where it changes your TR (following distance) dynamically based on multiple vehicle factors, as well as data from the lead vehicle. [Here's an old write up from a while ago explaining how it works exactly. Some of it might be out of date, but how it functions is the same.](https://github.com/ShaneSmiskol/openpilot/blob/dynamic-follow/README.md) The goal is to essentially make the driving experience more smooth and increase safety, braking and accelerating sooner.

Now you can choose a profile based on traffic and your driving preference. There are three profiles currently:
  * `traffic` - Meant to keep you a bit closer in traffic, hopefully reducing cut-ins. May not be the safest when approaching a vastly slower vehicle.
  * `relaxed` - This is the current and now default dynamic follow profile just with a cool name. Also slight closer than previously at high speeds.
  * `roadtrip` - This profile is for road trips mainly where you're on two lane highways and don't want to be following particularly closely; at night for example.
<img src=".media/photos/df_profiles.jpg?raw=true" width="800">

**How to choose a profile:** The easiest way is to use the new on-screen profile changing button! Once you're on a drive, you can simply tap the button on the bottom right of the screen to cycle between the profiles.

Dynamic gas
-----
Dynamic gas aims to provide a smoother driving experience in stop and go traffic (under 20 mph) by reducing the maximum gas that can be applied based on your current velocity, the relative velocity of the lead, the acceleration of the lead, and the distance of the lead. This usually results in quicker and smoother acceleration from a standstill without the jerking you get in stock openpilot with comma pedal (ex. taking off from a traffic light). It tries to coast if the lead is just inching up, it doesn’t use maximum gas as soon as the lead inches forward. When you are above 20 mph, relative velocity and the current following distance in seconds is taken into consideration.

All cars that have a comma pedal are supported! However to get the smoothest acceleration, I've custom tuned gas curve profiles for the following cars:

pedal cars:
  * 2017 Toyota Corolla (non-TSS2)
  * Toyota RAV4 (non-TSS2)
  * 2017 Honda Civic
  * 2019 Honda Pilot

non-pedal cars:
  * None yet

If you have a car without a pedal, or you do have one but I haven't created a profile for you yet, please let me know and we can develop one for your car to test.

~~*TODO: Need to factor in distance, as it will not accelerate to get closer to the stopped lead if you engage at ~0mph far back from the lead.*~~ Update: Added lead distance into dynamic gas, now just need to tune and test.

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
- `steer_ratio`: The steering ratio you want to use with openpilot. If you enter None, it will use the learned steer ratio from openpilot instead.
- `upload_on_hotspot`: Controls whether your EON will upload driving log data on your phone's hotspot
- `reset_integral`: Resets integral gain whenever the longitudinal PID error crosses or is zero. Helps overshoot

A list of parameters that you can modify are located [here](common/op_params.py#L42).

An archive of opParams [lives here.](https://github.com/ShaneSmiskol/op_params)

Parameters are stored at `/data/op_params.json`

Live tuning support
-----
Currently only the `camera_offset`, `lane_hug_angle_offset`, `dynamic_follow`, and `steer_ratio` parameters are supported.
- Just start opEdit with the instructions above and pick a parameter. It will let you know if it supports live tuning, if so, updates will take affect within 5 seconds!

<img src=".media/gifs/op_tune.gif?raw=true" width="600">

Dynamic lane speed
-----
*This feature is disabled until I can figure out how to improve it. Probably going to be replaced by comma's upcoming end 2 end long model.*

This is a feature that reduces your cruising speed if many vehicles around you are significantly slower than you. This works with and without an openpilot-identified lead. Ex.: It will slow you down if traveling in an open lane with cars in adjacent lanes that are slower than you. Or if the lead in front of the lead is slowing down, as well as cars in other lanes far ahead. The most it will slow you down is some average of: (the set speed and the average of the surrounding cars) The more the radar points, the more weight goes to the speeds of surrounding vehicles.

Custom wheel offset to reduce lane hugging
-----
***Update**: The performance of this modification is iffy at best, but it definitely is more apparent than just tuning your camera offset value. Removing the immediate angle offset can have some weird oscillating effect when it's windy or on roads with camber (slant to one side). Will be left in, but disabled by default.*

Stock openpilot doesn't seem to be able to identify your car's true angle offset. With the `LaneHugging` module you can specify a custom angle offset to be added to your desired steering angle. Simply find the angle your wheel is at when you're driving on a straight highway. By default, this is disabled, to enable you can:
- Use the `opEdit` class in the root directory of openpilot. To use it, simply open an `ssh` shell and enter the commands below:
    ```python
    cd /data/openpilot
    python op_edit.py
    ```
    You'll be greeted with a list of your parameters you can explore, enter the number corresponding to `lane_hug_direction`. Your options are to enter `'left'` or `'right'` for whichever direction your car has a tendency to hug toward. `None` will disable the feature.
    Finally you'll need to enter your absolute angle offset (negative will be converted to positive) with the `opParams` parameter: `lane_hug_angle_offset`.
    
    The lane hugging mod is enabled only if `lane_hug_direction` is `'left'` or `'right'`.

~~Two PID loops to control gas and brakes independently~~
-----
***Update**: Probably going to remove this addition, as tuning the current pedal parameters will be a more robust solution in the long run.*

If you have a Toyota Corolla with a comma pedal, you'll love this addition. Two longitudinal PID loops are set up in `longcontrol.py` so that one is running with comma pedal tuning to control the gas, and the other is running stock non-pedal tuning for better braking control. In the car, this feels miles better than stock openpilot, and nearly as good as your stock Toyota cruise control before you pulled out your DSU! It won't accelerate up to stopped cars and brake at the last moment anymore.

Automatic updates
-----
When a new update is available on GitHub for the `stock_additions` branch, your EON will pull and reset your local branch to the remote. It then queues a reboot to occur when the following is true:
- your EON has been inactive or offroad for more than 5 minutes.

Therefore, if the EON sees an update while you're driving it will reboot 5 minutes after you stop your drive, it resets the timer if you start driving again before the 5 minutes is up.

Documentation
=====

Quick Installation
-----
To install Stock Additions, just run the following on your EON/C2:

```
cd /data/
mv openpilot openpilot.old  # or equivalent
git clone -b stock_additions-release --single-branch https://github.com/shanesmiskol/openpilot
reboot
```

This branch is only 70MB enabling fast cloning of this fork! It will contain the latest from `stock_additions` squashed down to one commit. See below for more branch information.

Branches
-----
Most of the branches on this fork are development branches I use as various openpilot tests. The few that more permanent are the following:
  * [`stock_additions`](https://github.com/ShaneSmiskol/openpilot/tree/stock_additions): This is similar to stock openpilot's release branch. Will receive occasional and tested updates to Stock Additions.
  * [`stock_additions-devel`](https://github.com/ShaneSmiskol/openpilot/tree/stock_additions-devel): My development branch of Stock Additions I use to test new features or changes; similar to the master branch. Not recommendeded as a daily driver.
  * [`stock_additions-release`](https://github.com/ShaneSmiskol/openpilot/tree/stock_additions-release): The release-release branch sort of. Contains the latest of the `stock_additions` branch in one commit so cloning times are quick. 
  * [`traffic_lights-073`](https://github.com/ShaneSmiskol/openpilot/tree/traffic_lights-073):  The development branch testing our traffic light detection model [found here](https://github.com/ShaneSmiskol/traffic-lights).

The branches which have a version number appended to them are past archive versions of Stock Additions.
