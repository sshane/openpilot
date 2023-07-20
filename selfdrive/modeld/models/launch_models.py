import time
from selfdrive.manager.process_config import managed_processes

if __name__ == "__main__":
  while 1:
    managed_processes['navmodeld'].start()
    managed_processes['dmonitoringmodeld'].start()

    time.sleep(2)

    managed_processes['navmodeld'].stop()
    managed_processes['dmonitoringmodeld'].stop()
