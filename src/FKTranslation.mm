/*
 * fk-astro - XLIFF translation loader (see FKTranslation.h)
 */

#include "FKTranslation.h"

#include <dlfcn.h>
#include <stdio.h>

#include <Foundation/Foundation.h>

#include <map>
#include <string>

namespace fka
{

namespace
{

using TranslationMap = std::map<pcl::IsoString, pcl::IsoString>;
TranslationMap TheTranslations;
bool TheTranslationsLoaded = false;

// Directory that contains this module dylib.
std::string ModuleDirectory()
{
   Dl_info info;
   if ( ::dladdr( reinterpret_cast<void*>( &ModuleDirectory ), &info ) != 0
        && info.dli_fname != nullptr )
   {
      const char* end = ::strrchr( info.dli_fname, '/' );
      if ( end != nullptr )
         return std::string( info.dli_fname, end - info.dli_fname );
   }
   return std::string( "." );
}

std::string PreferredLanguage()
{
   if ( const char* env = ::getenv( "FKASTRO_LANG" ) )
      if ( *env != '\0' )
         return std::string( env );

   @autoreleasepool {
      NSString* lang = [NSLocale preferredLanguages].firstObject;
      if ( lang != nil )
         return std::string( lang.UTF8String ?: "en" );
   }
   return std::string( "en" );
}

// Resolve a system language tag to one of the shipped XLIFF files.
std::string LanguageCode()
{
   std::string lang = PreferredLanguage();
   if ( lang.rfind( "zh", 0 ) == 0 )
      return "zh-CN";   // zh-Hans-CN, zh_TW, ... all map to the single table
   return lang;
}

// Decode XML entities (&amp; &lt; &gt; &quot; &apos; &#NNN; &#xNN;) in place;
// numeric references are emitted as UTF-8. Good enough for the files we ship.
std::string UnescapeXML( const std::string& s )
{
   std::string r;
   r.reserve( s.length() );
   for ( size_t i = 0; i < s.length(); )
   {
      if ( s[i] == '&' )
      {
         size_t e = s.find( ';', i );
         if ( e != std::string::npos && e - i <= 12 )
         {
            std::string ent = s.substr( i + 1, e - i - 1 );
            if ( ent == "amp" ) { r += '&'; i = e + 1; continue; }
            if ( ent == "lt"  ) { r += '<'; i = e + 1; continue; }
            if ( ent == "gt"  ) { r += '>'; i = e + 1; continue; }
            if ( ent == "quot") { r += '"'; i = e + 1; continue; }
            if ( ent == "apos") { r += '\'';i = e + 1; continue; }
            if ( ent.length() > 1 && ent[0] == '#' )
            {
               unsigned cp = 0;
               bool ok = true;
               size_t k = 1;
               if ( ent[1] == 'x' || ent[1] == 'X' )
               {
                  k = 2;
                  for ( ; k < ent.length(); ++k )
                  {
                     char c = ent[k];
                     cp <<= 4;
                     if ( c >= '0' && c <= '9' ) cp |= unsigned( c - '0' );
                     else if ( c >= 'a' && c <= 'f' ) cp |= unsigned( c - 'a' + 10 );
                     else if ( c >= 'A' && c <= 'F' ) cp |= unsigned( c - 'A' + 10 );
                     else { ok = false; break; }
                  }
               }
               else
               {
                  for ( ; k < ent.length(); ++k )
                  {
                     char c = ent[k];
                     cp *= 10;
                     if ( c >= '0' && c <= '9' ) cp += unsigned( c - '0' );
                     else { ok = false; break; }
                  }
               }
               if ( ok )
               {
                  if ( cp < 0x80 )
                     r += char( cp );
                  else if ( cp < 0x800 )
                     r += char( 0xC0 | (cp >> 6) ),
                     r += char( 0x80 | (cp & 0x3F) );
                  else if ( cp < 0x10000 )
                     r += char( 0xE0 | (cp >> 12) ),
                     r += char( 0x80 | ((cp >> 6) & 0x3F) ),
                     r += char( 0x80 | (cp & 0x3F) );
                  else
                     r += char( 0xF0 | (cp >> 18) ),
                     r += char( 0x80 | ((cp >> 12) & 0x3F) ),
                     r += char( 0x80 | ((cp >> 6) & 0x3F) ),
                     r += char( 0x80 | (cp & 0x3F) );
                  i = e + 1;
                  continue;
               }
            }
         }
      }
      r += s[i++];
   }
   return r;
}

// Text content of the first <tag ...>...</tag> element inside block,
// entity-decoded. Returns false when the element is absent.
bool TagText( const std::string& block, const char* tag, std::string& text )
{
   const std::string open = std::string( "<" ) + tag;
   const std::string close = std::string( "</" ) + tag + ">";
   size_t b = block.find( open );
   while ( b != std::string::npos )
   {
      size_t e = block.find( '>', b );
      if ( e == std::string::npos )
         return false;
      if ( e > b && block[e-1] == '/' )   // self-closing <tag/>
      {
         b = block.find( open, e );
         continue;
      }
      size_t z = block.find( close, e );
      if ( z == std::string::npos )
         return false;
      text = UnescapeXML( block.substr( e + 1, z - e - 1 ) );
      return true;
   }
   return false;
}

// Parse <trans-unit> elements; the map key is the (unescaped) <source> text.
void LoadXLIFFFile( const std::string& path )
{
   FILE* f = ::fopen( path.c_str(), "rb" );
   if ( f == nullptr )
      return;
   std::string xml;
   char buffer[ 4096 ];
   size_t n;
   while ( (n = ::fread( buffer, 1, sizeof( buffer ), f )) > 0 )
      xml.append( buffer, n );
   ::fclose( f );

   size_t pos = 0;
   for ( ;; )
   {
      size_t b = xml.find( "<trans-unit", pos );
      if ( b == std::string::npos )
         break;
      size_t e = xml.find( "</trans-unit>", b );
      if ( e == std::string::npos )
         break;
      pos = e + 13;

      std::string source, target;
      if ( TagText( xml.substr( b, e - b ), "source", source )
          && TagText( xml.substr( b, e - b ), "target", target )
          && !source.empty() && !target.empty() )
         TheTranslations[pcl::IsoString( source.c_str() )]
            = pcl::IsoString( target.c_str() );
   }
}

} // namespace

void LoadTranslations()
{
   if ( TheTranslationsLoaded )
      return;
   TheTranslationsLoaded = true;

   const std::string code = LanguageCode();
   if ( code.empty() || code == "en" )
      return;   // source language: identity mapping

   LoadXLIFFFile( ModuleDirectory() + "/FKAstro-translations/" + code + ".xliff" );
}

pcl::String Tr( const char* s )
{
   TranslationMap::const_iterator i = TheTranslations.find( pcl::IsoString( s ) );
   if ( i != TheTranslations.end() )
      return pcl::String( i->second.UTF8ToUTF16() );
   return pcl::String( pcl::IsoString( s ).UTF8ToUTF16() );
}

} // namespace fka
