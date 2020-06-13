Stock Additions 0.7.5
=====

This branch is simply stock openpilot with some additions to help it drive as smooth as possible on my 2017 Toyota Corolla w/ comma pedal.

Want to request a feature or create a bug report? [Open an issue here!](https://github.com/ShaneSmiskol/openpilot/issues/new/choose)

If you'd like to support my development of Stock Additions with a [dollar for a RaceTrac ICEE](https://paypal.me/ssmiskol) (my PayPal link). Thanks! 🥰

-----

Highlight Features
=====

* [**Dynamic follow (now with profiles!)**](#dynamic-follow-3-profiles) - 3 + auto profiles to control distance
  * [**`auto-df` model for automatic distance profile switching**](#Automatic-DF-profile-switching)
* [**Dynamic gas**](#dynamic-gas) - smoother gas control
* [**PI > PID for longcontrol**](#Long-control-uses-a-PID-loop) - fix for pedal overshoot
* [**Customize this fork (opEdit with live tuning)**](#Customize-this-fork-opEdit)
* [**Automatic updates**](#Automatic-updates)

Documentation
=====
* [**Quick Installation**](#Quick-installation)
* [**Branches**](#Branches)
* [**Videos**](#Videos)

-----

Dynamic follow (3 profiles)
-----
Dynamic follow aims to provide the stock (Toyota) experience of having three different distance settings. Dynamic follow works by dynamically changing the distance in seconds which is sent to the long MPC to predict a speed to travel at. Basically, if the lead is decelerating or might soon, increase distance to prepare. And if the lead is accelerating, reduce distance to get up to speed quicker.

Dynamic follow works if openpilot can control your vehicle's gas and brakes (longitudinal). [Check if openpilot can control your vehicle's longitudinal from this list.](https://github.com/commaai/openpilot#supported-cars)

Just use the button on the button right of the screen while driving to change between these profiles:
  * [`traffic`](#Videos) - Meant to keep you a bit closer in traffic, hopefully reducing cut-ins. Always be alert, as you are with any driving assistance software.
  * `relaxed` - This is the default dynamic follow profile for casual driving.
  * `roadtrip` - This profile is for road trips where you're mainly on two lane highways and don't want to be following particularly closely; at night for example.
  * [`auto`](#Automatic-DF-profile-switching) - The auto dynamic follow model was trained on about an hour of me manually cycling through the different profiles based on driving conditions, this profile tries to replicate those decisions entirely on its own.

<img src=".media/df_profiles.jpg?raw=true" height="350">

Automatic DF profile switching
-----
I've trained a custom model with Keras that takes in the past 35 seconds of your speed, the lead's speed and the lead's distance. With these inputs, it tries to correctly predict which profile is the best for your current situation.

It's only been trained on about an hour of data, so it's not perfect yet, but it's great for users who just want to set it and forget it. **To enable the `auto` profile, simply tap the profile changing button for dynamic follow until it reaches the `auto` profile!**

If you're annoyed by the silent alerts that show when the model has changed the profile automatically, just use [opEdit](#Customize-this-fork-opEdit) and set `hide_auto_df_alerts` to `False`. Auto profile and model will remain functional but will not show alerts.

Resources:
- [The auto-df repo.](https://github.com/ShaneSmiskol/auto-df)
- [The model file.](https://github.com/ShaneSmiskol/openpilot/blob/stock_additions/selfdrive/controls/lib/dynamic_follow/auto_df.py)
- I converted the Keras model to be able to run with pure NumPy using [Konverter](https://github.com/ShaneSmiskol/Konverter).

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

Long control uses a PID loop
-----
I've added a custom implementation of derivative to the PI loop controlling the gas and brake output sent to your car. Derivative (change in error) is calculated based on the current and last error and added to the class's integral variable. It's essentially winding down integral according to derivative. It helps fix overshoot on some cars with the comma pedal and increases responsiveness (like when going up and down hills) on all other cars! Still need to figure out the tuning, right now it's using the same derivative gain for all cars. Test it out and let me know what you think!

Derivative is disabled by default due to only one tune for all cars, but can be enabled by using [opEdit](#Customize-this-fork-opEdit) and setting the `enable_long_derivative` parameter to `True`. It works well on my '17 Corolla with pedal.

Customize this fork (opEdit)
-----
This is a handy tool to change your `opParams` parameters without diving into any json files or code. You can specify parameters to be used in any fork's operation that supports `opParams`. First, ssh in to your EON and make sure you're in `/data/openpilot`, then start `opEdit`:
```python
cd /data/openpilot
python op_edit.py  # or ./op_edit.py
```

Features:
- You can misspell parameter names and opEdit should be able to figure out which parameter you want. Ex. `cmra off` would be parsed as: `camera_offset`
  - You can also still enter the corresponding parameter index while choosing parameters to edit
- Type `a` to add a parameter, `d` to delete a parameter, or `l` to toggle live tuning only mode
- Shows a detailed description for each parameter once you choose it
- Parameter value type restriction. Ensures a user cannot save an unsupported value type for any parameters, breaking the fork
- Remembers which mode you were last in and initializes opEdit with that mode (live tuning or not)
- Case insensitive boolean and NoneType entrance. Type `faLsE` to save `False (bool)`, etc
- **Parameters marked with `(live!)` will have updates take affect within 3 seconds while driving! All other params will require a reboot of your EON/C2 to take effect.**

Here are the main parameters you can change with this fork:
- **Tuning params**:
  - `camera_offset` **`(live!)`**: Your camera offset to use in lane_planner.py. Helps fix lane hugging
  - `steer_ratio` **`(live!)`**: The steering ratio you want to use with openpilot. If you enter None, it will use the learned steer ratio from openpilot instead
  - `enable_long_derivative`: This enables derivative-based integral wind-down to help overshooting within the PID loop. Useful for Toyotas with pedals or cars with bad long tuning
- **General fork params**:
  - `alca_nudge_required`: Whether to wait for applied torque to the wheel (nudge) before making lane changes
  - `alca_min_speed`: The minimum speed allowed for an automatic lane change
  - `upload_on_hotspot`: Controls whether your EON will upload driving data on your phone's hotspot
  - `no_ota_updates`: Set this to True to disable all automatic updates. Reboot to take effect
  - `disengage_on_gas`: Whether you want openpilot to disengage on gas input or not
- **Dynamic params**:
  - `dynamic_gas`: Whether to use [dynamic gas](#dynamic-gas) if your car is supported
  - `global_df_mod` **`(live!)`**: The modifer for the current distance used by dynamic follow. The range is limited from 0.7 to 1.1. Smaller values will get you closer, larger will get you farther. This is applied to ALL profiles!
  - `min_TR` **`(live!)`**: The minimum allowed following distance in seconds. Default is 0.9 seconds, the range of this mod is limited from 0.8 to 1.4 seconds. This is applied to ALL profiles!
  - `hide_auto_df_alerts`: Hides the alert that shows what profile the model has chosen
  - `dynamic_follow`: *Deprecated, use the on-screen button to change profiles*

A full list of parameters that you can modify are [located here](common/op_params.py#L40).

An archive of opParams [lives here.](https://github.com/ShaneSmiskol/op_params)

Parameters are stored at `/data/op_params.json`

## opEdit Demo
<img src=".media/op_edit.gif?raw=true" width="1000">

Automatic updates
-----
When a new update is available on GitHub for Stock Additions, your EON/C2 will pull and reset your local branch to the remote. It then queues a reboot to occur when the following is true:
- your EON has been inactive or offroad for more than 5 minutes.

Therefore, if your device sees an update while you're driving it will reboot approximately 5 to 10 minutes after you finish your drive, it resets the timer if you start driving again before the time is up.

Documentation
=====

Quick Installation
-----
To install Stock Additions, just run the following on your EON/C2 (make sure to press enter after each line):

```
cd /data/
mv openpilot openpilot.old  # or equivalent
git clone -b stock_additions --single-branch https://github.com/shanesmiskol/openpilot --depth 1
reboot
```

The `--depth 1` flag shallow clones the fork, it ends up being about 90 Mb so you can get the fork up and running quickly. Once you install Stock Additions, [automatic updating](#Automatic-updates) should always keep openpilot up to date with the latest from my fork!

Branches
-----
Most of the branches on this fork are development branches I use as various openpilot tests. The few that more permanent are the following:
  * [`stock_additions`](https://github.com/ShaneSmiskol/openpilot/tree/stock_additions): This is similar to stock openpilot's release branch. Will receive occasional and tested updates to Stock Additions.
  * [`stock_additions-devel`](https://github.com/ShaneSmiskol/openpilot/tree/stock_additions-devel): My development branch of Stock Additions I use to test new features or changes; similar to the master branch. Not recommendeded as a daily driver.
  * [`traffic_lights-074`](https://github.com/ShaneSmiskol/openpilot/tree/traffic_lights-074):  The development branch testing our traffic light detection model [found here](https://github.com/ShaneSmiskol/traffic-lights).

## Archive Stock Additions branches
* [Stock Additions 0.7](https://github.com/ShaneSmiskol/openpilot/tree/stock_additions-07)
* [Stock Additions 0.7.1](https://github.com/ShaneSmiskol/openpilot/tree/stock_additions-071)
* [Stock Additions 0.7.4](https://github.com/ShaneSmiskol/openpilot/tree/stock_additions-074)

Videos
-----
Here's a short video showing how close the traffic profile was in `0.7.4`. In `0.7.5`, the traffic profile is an average of 7.371 feet closer from 18 mph to 90 mph. Video thanks to [@rolo01](https://github.com/rolo01)!

[![](https://img.youtube.com/vi/sGsODeP_G_c/0.jpg)](https://www.youtube.com/watch?v=sGsODeP_G_c)
