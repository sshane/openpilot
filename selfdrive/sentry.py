"""Install exception handler for process crash."""
import sentry_sdk
from enum import Enum
from sentry_sdk.integrations.threading import ThreadingIntegration
import os
from datetime import datetime
import shutil
import traceback

from common.params import Params
from selfdrive.athena.registration import is_registered_device
from system.hardware import HARDWARE, PC
from system.swaglog import cloudlog
from system.version import get_branch, get_commit, get_origin, get_version, \
                              is_fork_remote, is_dirty, is_tested_branch

from common.op_params import opParams

CRASHES_DIR = '/data/community/crashes'


class SentryProject(Enum):
  # python project
  SELFDRIVE = "https://fde85a30a43a4a1b9d873c9b140143ac@o237581.ingest.sentry.io/6365324"
  # native project
  SELFDRIVE_NATIVE = "https://3e4b586ed21a4479ad5d85083b639bc6@o33823.ingest.sentry.io/157615"


def report_tombstone(fn: str, message: str, contents: str) -> None:
  cloudlog.error({'tombstone': message})

  with sentry_sdk.configure_scope() as scope:
    scope.set_extra("tombstone_fn", fn)
    scope.set_extra("tombstone", contents)
    sentry_sdk.capture_message(message=message)
    sentry_sdk.flush()


def capture_exception(*args, **kwargs) -> None:
  save_exception(traceback.format_exc())
  cloudlog.error("crash", exc_info=kwargs.get('exc_info', 1))

  try:
    sentry_sdk.capture_exception(*args, **kwargs)
    sentry_sdk.flush()  # https://github.com/getsentry/sentry-python/issues/291
  except Exception:
    cloudlog.exception("sentry exception")


def set_tag(key: str, value: str) -> None:
  sentry_sdk.set_tag(key, value)


def save_exception(exc_text):
  try:
    log_file = '{}/{}'.format(CRASHES_DIR, datetime.now().strftime('%Y-%m-%d--%H:%M.log'))
    with open(log_file, 'w') as f:
      f.write(exc_text)
    shutil.copyfile(log_file, '{}/latest.log'.format(CRASHES_DIR))
    print('Logged current crash to {} and {}'.format(log_file, '{}/latest.log'.format(CRASHES_DIR)))
  except:
    pass


def init(project: SentryProject) -> None:
  fork_remote = is_fork_remote() and "sshane" in get_origin(default="")
  # only report crashes to fork maintainer's sentry repo, skip native project
  if not fork_remote or not is_registered_device() or PC or project == SentryProject.SELFDRIVE_NATIVE:
    return

  if not os.path.exists(CRASHES_DIR):
    os.makedirs(CRASHES_DIR, exist_ok=True)

  env = "release" if is_tested_branch() else "master"
  dongle_id = Params().get("DongleId", encoding='utf-8')

  integrations = []
  if project == SentryProject.SELFDRIVE:
    integrations.append(ThreadingIntegration(propagate_hub=True))
  else:
    sentry_sdk.utils.MAX_STRING_LENGTH = 8192

  sentry_sdk.init(project.value,
                  default_integrations=False,
                  release=get_version(),
                  integrations=integrations,
                  traces_sample_rate=1.0,
                  environment=env)

  sentry_sdk.set_user({"id": dongle_id})
  sentry_sdk.set_tag("dirty", is_dirty())
  sentry_sdk.set_tag("origin", get_origin())
  sentry_sdk.set_tag("branch", get_branch())
  sentry_sdk.set_tag("commit", get_commit())
  sentry_sdk.set_tag("device", HARDWARE.get_device_type())
  sentry_sdk.set_tag("username", opParams().get('username'))

  if project == SentryProject.SELFDRIVE:
    sentry_sdk.Hub.current.start_session()
