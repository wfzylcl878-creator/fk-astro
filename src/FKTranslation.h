#pragma once

/*
 * fk-astro - XLIFF-based interface localization.
 *
 * PCL 2.11 exposes no localization framework to modules, so this is a small
 * self-contained mechanism: standard XLIFF 1.2 files are shipped next to the
 * module dylib in FKAstro-translations/<lang>.xliff and parsed once at GUI
 * creation. Tr() maps interface source strings (English, ASCII) to their
 * translated equivalents; untranslated strings pass through unchanged.
 *
 * Language selection: FKASTRO_LANG environment variable overrides, then the
 * macOS preferred system language ("zh*" -> zh-CN).
 */

#include <pcl/String.h>

namespace fka
{

// Load the translation table once (idempotent, cheap when already loaded).
void LoadTranslations();

// Translate an interface string. Returns the UTF-16 target text from the
// loaded XLIFF table, or the source string unchanged when no table or no
// entry exists.
pcl::String Tr( const char* s );

} // namespace fka
