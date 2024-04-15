include ../Rules.mk

APP := 04_p210-cuda

SRCS := \
	main.cpp \
	kernels.cu \
	$(wildcard $(COMMON_DIR)/*.cpp)

OBJS := $(filter %.o,$(SRCS:.cpp=.o) $(SRCS:.cu=.o))

CXXFLAGS += -I/usr/local/cuda/include

LDFLAGS += -Wl,-Bstatic
LDFLAGS += -lboost_filesystem -lboost_system

LDFLAGS += -Wl,-Bdynamic
LDFLAGS += -L/usr/local/cuda/lib64
LDFLAGS += -lcuda -lcudart

all: $(APP)

clean:
	$(AT)rm -rf $(APP) $(OBJS)

$(COMMON_DIR)/%.o: $(COMMON_DIR)/%.cpp
	$(AT)$(MAKE) -C $(COMMON_DIR)

%.o: %.cpp
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

%.o: %.cu
	@echo "CUDA Compiling: $<"
	${NVCC} $(GENCODE_FLAGS) -c $< -o $@

$(APP): $(OBJS)
	@echo "Linking: $@"
	$(CXX) -o $@ $(OBJS) $(CXXFLAGS) $(LDFLAGS)
