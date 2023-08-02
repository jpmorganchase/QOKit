CC = gcc
CFLAGS = -fopenmp -fPIC -Wall
LDFLAGS = -fopenmp -shared

VERSION_SCRIPT = libcsim.map

INC_DIRS := .
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

TARGET_DIR := ..

SRCS := $(wildcard *.c)
OBJS := $(SRCS:.c=.o)
TARGET := $(TARGET_DIR)/libcsim.so


all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -Wl,--version-script=$(VERSION_SCRIPT) $(OBJS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INC_FLAGS) -c $< -o $@


clean:
	$(RM) $(OBJS)
	$(RM) $(TARGET)
