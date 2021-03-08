CC = clang
CXX = clang++

ROOT_DIR = ../../../..
PHONELIBS = $(ROOT_DIR)/phonelibs
COMMON = $(ROOT_DIR)/selfdrive/common

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++1z -g -fPIC -O2 $(WARN_FLAGS)

NANOVG_FLAGS = -I$(PHONELIBS)/nanovg

OPENGL_LIBS = -lGLESv3

FRAMEBUFFER_LIBS = -lutils -lgui -lEGL

OBJS = spinner.o \
			 opensans_semibold.ttf.o \
			 opensans_bold.ttf.o \
       $(COMMON)/util.o \
	   $(COMMON)/framebuffer.o \
       $(PHONELIBS)/nanovg/nanovg.o \
       opensans_semibold.o \
       img_spinner_track.o \
       img_spinner_comma.o

DEPS := $(OBJS:.o=.d)

.PHONY: all
all: spinner

spinner: $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
	      -s \
        $(FRAMEBUFFER_LIBS) \
	    -I$(ROOT_DIR)/selfdrive/common \
        -L/system/vendor/lib64 \
        $(OPENGL_LIBS) \
        -lm -llog


$(COMMON)/framebuffer.o: $(COMMON)/framebuffer.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
		   -I$(ROOT_DIR)/selfdrive/common \
           -I$(PHONELIBS)/android_frameworks_native/include \
           -I$(PHONELIBS)/android_system_core/include \
           -I$(PHONELIBS)/android_hardware_libhardware/include \
           -c -o '$@' '$<'


opensans_semibold.o: $(ROOT_DIR)/selfdrive/assets/fonts/opensans_semibold.ttf
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

img_spinner_track.o: $(ROOT_DIR)/selfdrive/assets/img_spinner_track.png
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

img_spinner_comma.o: $(ROOT_DIR)/selfdrive/assets/img_spinner_comma.png
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -I$(ROOT_DIR)/selfdrive/common \
          -I$(ROOT_DIR)/selfdrive \
          -I$(ROOT_DIR) \
          -I$(PHONELIBS)/android_frameworks_native/include \
          -I$(PHONELIBS)/android_system_core/include \
          -I$(PHONELIBS)/android_hardware_libhardware/include \
          $(NANOVG_FLAGS) \
          -c -o '$@' '$<'

%.o: %.cc
	mkdir -p $(@D)
	@echo "[ CXX ] $@"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
           -I$(ROOT_DIR)/selfdrive/common \
           -I$(ROOT_DIR)/selfdrive \
           -I$(ROOT_DIR) \
           -I$(PHONELIBS)/android_frameworks_native/include \
           -I$(PHONELIBS)/android_system_core/include \
           -I$(PHONELIBS)/android_hardware_libhardware/include \
           $(NANOVG_FLAGS) \
           -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f spinner $(OBJS) $(DEPS)

-include $(DEPS)
