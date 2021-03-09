import time
from common.spinner import Spinner
import sys

spinner = Spinner()
spinner.update_progress(0, 100)

argv = sys.argv[1:]
rate = 1 / float(argv[0])

for i in range(100):
  print(i)
  spinner.update_progress(i, 100, rate)
  # time.sleep(rate)
time.sleep(5)
spinner.close()
