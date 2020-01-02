from cffi import FFI
import subprocess
try:
    subprocess.check_call(["make", "-j4"], cwd="/data/openpilot/selfdrive/smart_torque")
except:
    pass


def get_wrapper():  # initialize st model and process long predictions
    libmpc_fn = "/data/openpilot/selfdrive/smart_torque/smart_torque.so"

    ffi = FFI()
    ffi.cdef("""    
    void init_model();
    float run_model(float inputData[1000]);
    """)

    return ffi.dlopen(libmpc_fn)