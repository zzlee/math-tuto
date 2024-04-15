include ../Rules.mk

SRCS:= $(wildcard *.cpp)

OBJS := $(SRCS:.cpp=.o)

all: $(OBJS)

%.o: %.cpp
	@echo "Compiling: $<"
	$(CPP) $(CPPFLAGS) -fPIC -c $< -o $@

clean:
	$(AT)rm -rf $(OBJS)
