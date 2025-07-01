# ─────────────────────────  build options  ──────────────────────────────
CC      ?= gcc            # allows: make CC=gcc-15 (Homebrew GCC-15 with OMP)
CFLAGS  = -fopenmp -fPIC -Wall
LDFLAGS = -fopenmp -shared

# Linux-only version-script
VERSION_SCRIPT = libcsim.map
ifeq ($(shell uname),Linux)
    LDFLAGS += -Wl,--version-script=$(VERSION_SCRIPT)
endif

INC_DIRS  := .
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

TARGET_DIR := ..
SRCS   := $(wildcard *.c)
OBJS   := $(SRCS:.c=.o)
TARGET := $(TARGET_DIR)/libcsim.so

# ─────────────────────────  build rules  ────────────────────────────────
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
