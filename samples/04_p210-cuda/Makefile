include ../Rules.mk

APP := 04_p210-cuda

SRCS := \
	main.cpp \
	kernels.cu \
	$(wildcard $(COMMON_DIR)/*.cpp)

OBJS := $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

$(info OBJS=${OBJS})

LDFLAGS += -lboost_filesystem
LDFLAGS += -L/usr/local/cuda/targets/aarch64-linux/lib
LDFLAGS += -lcuda -lcudart

all: $(APP)

clean:
	$(AT)rm -rf $(APP) $(OBJS)

$(COMMON_DIR)/%.o: $(COMMON_DIR)/%.cpp
	$(AT)$(MAKE) -C $(COMMON_DIR)

%.o: %.cpp
	@echo "Compiling: $<"
	$(CPP) $(CPPFLAGS) -fPIC -c $< -o $@

%.o: %.cu
	@echo "CUDA Compiling: $<"
	${NVCC} $(GENCODE_FLAGS) -c $< -o $@

$(APP): $(OBJS)
	@echo "Linking: $@ ($^)"
	$(CPP) -o $@ $(OBJS) $(CPPFLAGS) $(LDFLAGS)
