include ../Rules.mk

APP := 03_p210

SRCS := \
	main.cpp \
	$(wildcard $(COMMON_DIR)/*.cpp)

OBJS := $(SRCS:.cpp=.o)

LDFLAGS += -Wl,-Bstatic
LDFLAGS += -lboost_filesystem -lboost_system

LDFLAGS += -Wl,-Bdynamic

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
