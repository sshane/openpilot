import time
from common.spinner import Spinner

spinner = Spinner()
spinner.update_progress(0, 100)


for i in range(1000):
  spinner.update_progress(i, 1000)
  time.sleep(1/1000)
time.sleep(5)
spinner.close()