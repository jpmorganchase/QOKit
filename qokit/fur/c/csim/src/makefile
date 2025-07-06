# ─────────────────────────  build options  ──────────────────────────────
CC      ?= gcc
CFLAGS  = -fopenmp -fPIC -Wall
LDFLAGS = -fopenmp -shared
TARGET_DIR := ..
SRCS   := $(wildcard *.c)
OBJS   := $(SRCS:.c=.o)
TARGET := $(TARGET_DIR)/libcsim.so

# Linux-only version script
ifeq ($(shell uname),Linux)
    LDFLAGS += -Wl,--version-script=libcsim.map
endif

INC_DIRS  := .
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# ─────────────────────────  build rules  ────────────────────────────────
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
