import os

from selfdrive.manager.process import PythonProcess, NativeProcess, DaemonProcess
from selfdrive.hardware import EON, TICI, PC
from common.op_params import opParams

WEBCAM = os.getenv("USE_WEBCAM") is not None

procs = [
  DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid"),
  # due to qualcomm kernel bugs SIGKILLing camerad sometimes causes page table corruption
  NativeProcess("camerad", "selfdrive/camerad", ["./camerad"], unkillable=True, driverview=True, sentry=True),
  NativeProcess("clocksd", "selfdrive/clocksd", ["./clocksd"], sentry=True),
  NativeProcess("dmonitoringmodeld", "selfdrive/modeld", ["./dmonitoringmodeld"], enabled=(not PC or WEBCAM), driverview=True),
  NativeProcess("logcatd", "selfdrive/logcatd", ["./logcatd"], sentry=True),
  NativeProcess("loggerd", "selfdrive/loggerd", ["./loggerd"], sentry=True),
  NativeProcess("modeld", "selfdrive/modeld", ["./modeld"], sentry=True),
  NativeProcess("navd", "selfdrive/ui/navd", ["./navd"], enabled=(PC or TICI), persistent=True),
  NativeProcess("proclogd", "selfdrive/proclogd", ["./proclogd"], sentry=True),
  NativeProcess("sensord", "selfdrive/sensord", ["./sensord"], enabled=not PC, persistent=True, sigkill=EON),
  NativeProcess("ubloxd", "selfdrive/locationd", ["./ubloxd"], enabled=(not PC or WEBCAM)),
  NativeProcess("ui", "selfdrive/ui", ["./ui"], persistent=True, watchdog_max_dt=(5 if TICI else None)),
  NativeProcess("soundd", "selfdrive/ui/soundd", ["./soundd"], persistent=True),
  NativeProcess("locationd", "selfdrive/locationd", ["./locationd"]),
  NativeProcess("boardd", "selfdrive/boardd", ["./boardd"], enabled=False),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd", sentry=True),  # TODO: so UI will show. fix
  PythonProcess("controlsd", "selfdrive.controls.controlsd"),
  PythonProcess("deleter", "selfdrive.loggerd.deleter", persistent=True),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", enabled=(not PC or WEBCAM), driverview=True),
  PythonProcess("lanespeedd", "selfdrive.controls.lane_speed"),
  PythonProcess("logmessaged", "selfdrive.logmessaged", persistent=True),
  PythonProcess("pandad", "selfdrive.pandad", persistent=True),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd"),
  PythonProcess("plannerd", "selfdrive.controls.plannerd"),
  PythonProcess("radard", "selfdrive.controls.radard"),
  PythonProcess("sentryd", "selfdrive.sentryd", persistent=True),
  PythonProcess("thermald", "selfdrive.thermald.thermald", persistent=True),
  PythonProcess("timezoned", "selfdrive.timezoned", enabled=TICI, persistent=True),
  PythonProcess("tombstoned", "selfdrive.tombstoned", enabled=not PC, persistent=True),
  PythonProcess("uploader", "selfdrive.loggerd.uploader", persistent=True),

  # EON only
  PythonProcess("rtshield", "selfdrive.rtshield", enabled=EON),
  PythonProcess("androidd", "selfdrive.hardware.eon.androidd", enabled=EON, persistent=True),
]

if opParams().get('update_behavior').lower().strip() != 'off' and not os.path.exists('/data/no_ota_updates'):
  procs.append(PythonProcess("updated", "selfdrive.updated", enabled=not PC, persistent=True))

managed_processes = {p.name: p for p in procs}
