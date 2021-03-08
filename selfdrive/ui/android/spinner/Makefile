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

all: spinner

OBJS = spinner.o \
	         opensans_regular.ttf.o \
			 opensans_semibold.ttf.o \
			 opensans_bold.ttf.o \
       $(COMMON)/util.o \
       $(COMMON)/touch.o \
       $(COMMON)/framebuffer.o \
       $(PHONELIBS)/nanovg/nanovg.o \
       $(COMMON)/spinner.o \
       img_spinner_track.o \
       img_spinner_comma.o

DEPS := $(OBJS:.o=.d)

.PHONY: all
all: spinner

spinner: spinner.o $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
        $(FRAMEBUFFER_LIBS) \
        -L/system/vendor/lib64 \
        $(OPENGL_LIBS) \
        -lcutils -lm -llog \
        -lm -llog

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
          -I../../.. \
          -I../.. \
          -I../../../.. \
          -I$(PHONELIBS)/android_frameworks_native/include \
          -I$(PHONELIBS)/android_system_core/include \
          -I$(PHONELIBS)/android_hardware_libhardware/include \
          $(NANOVG_FLAGS) \
          -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f spinner $(OBJS) $(DEPS)

-include $(DEPS)
