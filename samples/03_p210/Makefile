include ../Rules.mk

APP := 03_p210

SRCS := \
	main.cpp \
	$(wildcard $(COMMON_DIR)/*.cpp)

OBJS := $(SRCS:.cpp=.o)

LDFLAGS += -lboost_filesystem

all: $(APP)

clean:
	$(AT)rm -rf $(APP) $(OBJS)

$(COMMON_DIR)/%.o: $(COMMON_DIR)/%.cpp
	$(AT)$(MAKE) -C $(COMMON_DIR)

%.o: %.cpp
	@echo "Compiling: $<"
	$(CPP) $(CPPFLAGS) -fPIC -c $< -o $@

$(APP): $(OBJS)
	@echo "Linking: $@"
	$(CPP) -o $@ $(OBJS) $(CPPFLAGS) $(LDFLAGS)
