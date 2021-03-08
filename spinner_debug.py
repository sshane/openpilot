import time
from common.spinner import Spinner

spinner = Spinner()
spinner.update_progress(0, 100)


for i in range(100):
  print(i)
  spinner.update_progress(i, 100)
  time.sleep(1/100)
time.sleep(5)
spinner.close()
