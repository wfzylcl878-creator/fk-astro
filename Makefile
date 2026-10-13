######################################################################
# fk-astro — build system
#
# Builds a PixInsight module (universal arm64 + x86_64 when the installed
# PCL supports arm64, x86_64 only otherwise):
#
#   FKAstro-pxm.dylib
#
# Prerequisites:
#   - PixInsight installation (PCL headers + sources) at $(PI)
#   - Xcode with MacOSX SDK
#
# Targets:
#   make          — build everything (third-party, PCL, module)
#   make module   — same as above
#   make pcl      — build libPCL-pxi.a only
#   make clean    — remove build artifacts
######################################################################

PI      ?= /Applications/PixInsight
PCLSRC   = $(PI)/src
PCLINC   = $(PI)/include
T        = $(PCLSRC)/3rdparty
SDK      = $(shell xcrun --sdk macosx --show-sdk-path)
CXX      = clang++
CC       = clang
AR       = ar
# Build arm64 only when the installed PCL supports it (Apple Silicon
# PixInsight builds ship makefile-arm64; Intel-only builds do not).
ARCHS   ?= $(if $(wildcard $(PCLSRC)/pcl/macosx/g++/makefile-arm64),arm64 x64,x64)
# clang arch flag per build dir ("x64" is PCL naming; clang wants x86_64)
ARCH_CLANG_arm64 = arm64
ARCH_CLANG_x64 = x86_64
DIST     = dist

#
# Flags replicated verbatim from the official PixInsight PCL makefiles:
# $(PCLSRC)/pcl/macosx/g++/makefile-{arm64,x64}
#
BASE_CXXFLAGS = -pipe -pthread -fPIC -isysroot "$(SDK)" -mmacosx-version-min=14 \
   -D_REENTRANT -D__PCL_MACOSX -I"$(PCLINC)" -I"$(T)" \
   -minline-all-stringops -O3 -ffunction-sections -fdata-sections -ffast-math \
   -ffp-contract=on -Wno-overriding-option -fvisibility=hidden \
   -fvisibility-inlines-hidden -std=c++20 -stdlib=libc++ \
   -Wall -Wno-parentheses -Wno-missing-braces -Wno-nan-infinity-disabled -Wno-extern-c-compat

BASE_CFLAGS = -pipe -O2 -fPIC -isysroot "$(SDK)" -mmacosx-version-min=14

# --- module sources -------------------------------------------------

MODULE_CPP := $(wildcard src/*.cpp)
MODULE_MM  := $(wildcard src/*.mm)

# --- PCL sources ----------------------------------------------------

PCL_CPP := $(notdir $(wildcard $(PCLSRC)/pcl/*.cpp))
PCL_MM  := $(notdir $(wildcard $(PCLSRC)/pcl/*.mm))

# --- third-party sources --------------------------------------------

ZLIB_C    := $(wildcard $(T)/zlib/*.c)
LZ4_C     := $(T)/lz4/lz4.c $(T)/lz4/lz4hc.c
ZSTD_C    := $(wildcard $(T)/zstd/common/*.c $(T)/zstd/compress/*.c $(T)/zstd/decompress/*.c $(T)/zstd/*.c)
RFC_C     := $(wildcard $(T)/RFC6234/*.c)
LCMS_C    := $(wildcard $(T)/lcms/*.c)

######################################################################
# Rules
######################################################################

.PHONY: all module pcl clean

all: $(DIST)/FKAstro-pxm.dylib
module: $(DIST)/FKAstro-pxm.dylib

$(DIST)/FKAstro-pxm.dylib: $(foreach A,$(ARCHS),build/$(A)/FKAstro-pxm.dylib)
	@mkdir -p $(DIST)
	lipo -create -output $@ $^
	codesign -f -s - $@
	@echo "=== Built $@ ==="
	@echo "Next: sign it as a PixInsight module (see README.md) and copy it to $(PI)/bin/"

#
# Third-party static libraries (one per architecture)
#
define THIRD_PARTY_RULES

build/$(1)/libzlib-pxi.a: $$(patsubst $(T)/%.c,build/$(1)/3rdparty/%.o,$$(ZLIB_C))
	@mkdir -p $$(@D)
	$$(AR) r $$@ $$^

build/$(1)/liblz4-pxi.a: $$(patsubst $(T)/%.c,build/$(1)/3rdparty/%.o,$$(LZ4_C))
	@mkdir -p $$(@D)
	$$(AR) r $$@ $$^

build/$(1)/libzstd-pxi.a: $$(patsubst $(T)/%.c,build/$(1)/3rdparty/%.o,$$(ZSTD_C))
	@mkdir -p $$(@D)
	$$(AR) r $$@ $$^

build/$(1)/libRFC6234-pxi.a: $$(patsubst $(T)/%.c,build/$(1)/3rdparty/%.o,$$(RFC_C))
	@mkdir -p $$(@D)
	$$(AR) r $$@ $$^

build/$(1)/liblcms-pxi.a: $$(patsubst $(T)/%.c,build/$(1)/3rdparty/%.o,$$(LCMS_C))
	@mkdir -p $$(@D)
	$$(AR) r $$@ $$^

build/$(1)/3rdparty/zstd/%.o: $(T)/zstd/%.c
	@mkdir -p $$(@D)
	$$(CC) $$(BASE_CFLAGS) -arch $$(ARCH_CLANG_$(1)) -I"$(T)/zstd" -c -o $$@ $$<

build/$(1)/3rdparty/lz4/%.o: $(T)/lz4/%.c
	@mkdir -p $$(@D)
	$$(CC) $$(BASE_CFLAGS) -arch $$(ARCH_CLANG_$(1)) -I"$(T)/lz4" -c -o $$@ $$<

build/$(1)/3rdparty/zlib/%.o: $(T)/zlib/%.c
	@mkdir -p $$(@D)
	$$(CC) $$(BASE_CFLAGS) -arch $$(ARCH_CLANG_$(1)) -DZ_HAVE_UNISTD_H -Dfdopen=fdopen -I"$(T)/zlib" -c -o $$@ $$<

build/$(1)/3rdparty/RFC6234/%.o: $(T)/RFC6234/%.c
	@mkdir -p $$(@D)
	$$(CC) $$(BASE_CFLAGS) -arch $$(ARCH_CLANG_$(1)) -I"$(T)/RFC6234" -c -o $$@ $$<

build/$(1)/3rdparty/lcms/%.o: $(T)/lcms/%.c
	@mkdir -p $$(@D)
	$$(CC) $$(BASE_CFLAGS) -arch $$(ARCH_CLANG_$(1)) -I"$(T)/lcms" -c -o $$@ $$<

endef

$(foreach A,$(ARCHS),$(eval $(call THIRD_PARTY_RULES,$(A))))

#
# PCL static library
#
define PCL_RULES

PCL_OBJS_$(1) := $$(addprefix build/$(1)/pcl/,$$(PCL_CPP:.cpp=.o) $$(PCL_MM:.mm=.o))

build/$(1)/libPCL-pxi.a: $$(PCL_OBJS_$(1))
	@mkdir -p $$(@D)
	$$(AR) r $$@ $$^

build/$(1)/pcl/%.o: $(PCLSRC)/pcl/%.cpp
	@mkdir -p $$(@D)
	$$(CXX) $$(BASE_CXXFLAGS) -arch $$(ARCH_CLANG_$(1)) -c -o $$@ $$<

build/$(1)/pcl/%.o: $(PCLSRC)/pcl/%.mm
	@mkdir -p $$(@D)
	$$(CXX) $$(BASE_CXXFLAGS) -arch $$(ARCH_CLANG_$(1)) -c -o $$@ $$<

pcl-$(1): build/$(1)/libPCL-pxi.a

endef

$(foreach A,$(ARCHS),$(eval $(call PCL_RULES,$(A))))

pcl: $(foreach A,$(ARCHS),pcl-$(A))

#
# Module
#
define MODULE_RULES

MODULE_OBJS_$(1) := $$(addprefix build/$(1)/module/,$$(MODULE_CPP:src/%.cpp=%.o) $$(MODULE_MM:src/%.mm=%.o))

build/$(1)/FKAstro-pxm.dylib: $$(MODULE_OBJS_$(1)) \
      build/$(1)/libPCL-pxi.a \
      build/$(1)/libzlib-pxi.a build/$(1)/liblz4-pxi.a build/$(1)/libzstd-pxi.a \
      build/$(1)/libRFC6234-pxi.a build/$(1)/liblcms-pxi.a
	$$(CXX) -arch $$(ARCH_CLANG_$(1)) -fPIC -headerpad_max_install_names \
	   -Wl,-syslibroot,"$$(SDK)" -mmacosx-version-min=14 -stdlib=libc++ -Wl,-dead_strip \
	   -dynamiclib -install_name @executable_path/FKAstro-pxm.dylib \
	   -Lbuild/$(1) -o $$@ $$(MODULE_OBJS_$(1)) \
	   -framework AppKit -framework Foundation -framework CoreML \
	   -lpthread -lPCL-pxi -llz4-pxi -lzstd-pxi -lzlib-pxi -lRFC6234-pxi -llcms-pxi

build/$(1)/module/%.o: src/%.cpp
	@mkdir -p $$(@D)
	$$(CXX) $$(BASE_CXXFLAGS) -Isrc -arch $$(ARCH_CLANG_$(1)) -c -o $$@ $$<

build/$(1)/module/%.o: src/%.mm
	@mkdir -p $$(@D)
	$$(CXX) $$(BASE_CXXFLAGS) -Isrc -arch $$(ARCH_CLANG_$(1)) -fobjc-arc -fobjc-arc-exceptions -c -o $$@ $$<

endef

$(foreach A,$(ARCHS),$(eval $(call MODULE_RULES,$(A))))

clean:
	rm -rf build dist
