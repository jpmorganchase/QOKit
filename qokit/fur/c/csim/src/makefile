# ─────────────────────────  build options  ──────────────────────────────
CC      ?= gcc
CFLAGS  = -fopenmp -fPIC -Wall
TARGET_DIR := ..
SRCS   := $(wildcard *.c)
OBJS   := $(SRCS:.c=.o)

# Detect platform
UNAME_S := $(shell uname -s)

# Default values (Linux)
LDFLAGS = -fopenmp -shared
TARGET = $(TARGET_DIR)/libcsim.so

# macOS-specific settings
ifeq ($(UNAME_S),Darwin)
    LDFLAGS = -fopenmp -dynamiclib \
              -Wl,-syslibroot,$(shell xcrun --show-sdk-path) \
              -install_name @rpath/libcsim.dylib
    TARGET = $(TARGET_DIR)/libcsim.dylib
endif

# Linux-only version script
ifeq ($(UNAME_S),Linux)
    LDFLAGS += -Wl,--version-script=libcsim.map
endif

INC_DIRS  := .
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# ─────────────────────────  build rules  ────────────────────────────────
all: $(TARGET)
ifeq ($(UNAME_S),Darwin)
	# Copy .dylib to .so for Python compatibility
	cp -f $(TARGET_DIR)/libcsim.dylib $(TARGET_DIR)/libcsim.so
endif

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET_DIR)/libcsim.so $(TARGET_DIR)/libcsim.dylib
