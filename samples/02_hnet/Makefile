include ../Rules.mk

APP := 02_hnet

SRCS := \
	main.cpp \
	$(wildcard $(COMMON_DIR)/*.cpp)

OBJS := $(SRCS:.cpp=.o)

all: $(APP)

clean:
	$(AT)rm -rf $(APP) $(OBJS)

$(COMMON_DIR)/%.o: $(COMMON_DIR)/%.cpp
	$(AT)$(MAKE) -C $(COMMON_DIR)

%.o: %.cpp
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

$(APP): $(OBJS)
	@echo "Linking: $@"
	$(CXX) -o $@ $(OBJS) $(CXXFLAGS) $(LDFLAGS)
