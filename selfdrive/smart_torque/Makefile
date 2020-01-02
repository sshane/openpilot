CXX = clang++

PHONELIBS = ../../phonelibs

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args \
             -Wno-deprecated-declarations

CFLAGS = -std=gnu11 -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++14 -fPIC -O2 $(WARN_FLAGS)

SNPE_FLAGS = -I$(PHONELIBS)/snpe/include/
SNPE_LIBS = -lSNPE -lsymphony-cpu -lsymphonypower

OBJS = smart_torque.o
OUTPUT = smart_torque.so

.PHONY: all
all: $(OUTPUT)

DEPS := $(OBJS:.o=.d)

$(OUTPUT): $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
        $(LDFLAGS) \
        $(SNPE_LIBS) \
        -shared

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
           -Iinclude -I.. -I../.. \
           $(SNPE_FLAGS) \
           -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f $(OBJS) $(DEPS)