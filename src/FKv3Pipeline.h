#pragma once

// FKv3 host pipeline, based on the ML5 contract recovered from
// RCAstroSuite-pxm.dylib (ARM64).
// This file has no PCL/CoreML dependency so the production math can be tested
// offline. See docs/FKv3.md for evidence addresses and verification limits.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace fka
{
namespace fkv3
{

template <class S>
double SampleToDouble( S value )
{
   if constexpr ( std::is_floating_point<S>::value )
      return double( value );
   else
      return double( value ) / double( std::numeric_limits<S>::max() );
}

template <class S>
S SampleFromDouble( double value )
{
   if constexpr ( std::is_floating_point<S>::value )
      return S( value );
   else
      // The host multiplies by the integer maximum, then uses the PCL
      // nearest/even conversion (default IEEE rounding), not truncation.
      return S( std::nearbyint( value * double( std::numeric_limits<S>::max() ) ) );
}

struct Settings
{
   double stars = 0.5, halos = 0, diameter = 0, nonstellar = 0.5;
   bool autoPSF = true, correctOnly = false, lunar = false;

   Settings Effective() const
   {
      Settings result = *this;
      if ( correctOnly )
      {
         result.stars = result.halos = result.diameter = result.nonstellar = 0;
         result.autoPSF = true;
         result.lunar = false;
      }
      if ( result.lunar )
      {
         result.stars = result.halos = 0;
         result.autoPSF = false;
      }
      if ( result.autoPSF )
         result.diameter = 0;
      return result;
   }

   std::array<float, 6> Parameters( double gain ) const
   {
      const Settings e = Effective();
      return { float( e.stars ), float( e.halos ), float( e.diameter * 0.5 ),
               float( e.autoPSF ), float( e.nonstellar ), float( gain ) };
   }
};

struct ArcsinhNormalizer
{
   double low = 0, high = 1, median = 0, x = 1;
   double asinhX = std::asinh( 1.0 ), gain = std::log10( 1.0 / std::asinh( 1.0 ) );
   int cropX = 0, cropY = 0, cropW = 0, cropH = 0, gridW = 0, gridH = 0;
   size_t positiveSamples = 0;

   // at(c,y,x) reads the original sample as double. Fit once for the whole
   // image; reuse this exact state for input pixels, params[5] and inverse.
   template <class PixelAt>
   void Fit( int width, int height, int channels, bool lunar, PixelAt at )
   {
      *this = ArcsinhNormalizer();
      if ( width < 1 || height < 1 || channels < 1 )
         throw std::invalid_argument( "FKv3 normalization requires a nonempty image" );
      cropX = int( width * 0.05 );
      cropY = int( height * 0.05 );
      cropW = width - 2 * cropX;
      cropH = height - 2 * cropY;
      if ( cropW < 1 || cropH < 1 )
      {
         cropX = cropY = 0;
         cropW = width;
         cropH = height;
      }
      const double scale = double( std::max( cropW, cropH ) ) / 128.0;
      gridW = std::max( 1, int( std::llround( cropW / scale ) ) );
      gridH = std::max( 1, int( std::llround( cropH / scale ) ) );
      const double stepX = double( cropW ) / gridW, stepY = double( cropH ) / gridH;
      std::vector<double> samples;
      samples.reserve( size_t( gridW ) * gridH * channels );
      for ( int c = 0; c < channels; ++c )
         for ( int gy = 0; gy < gridH; ++gy )
         {
            const int sy = cropY + std::min( cropH - 1, int( std::floor( gy * stepY ) ) );
            for ( int gx = 0; gx < gridW; ++gx )
            {
               const int sx = cropX + std::min( cropW - 1, int( std::floor( gx * stepX ) ) );
               const double value = at( c, sy, sx );
               if ( value > 0 )
                  samples.push_back( value );
            }
         }
      positiveSamples = samples.size();
      // high remains 1: the ML5 catalog specifies arcsinh,wp=1.
      if ( !samples.empty() )
      {
         std::sort( samples.begin(), samples.end() );
         const double q = double( samples.size() - 1 ) * 0.001;
         const size_t i = size_t( q );
         low = std::fma( q - double( i ), samples[std::min( i + 1, samples.size() - 1 )] - samples[i], samples[i] );
      }
      if ( std::fabs( high - low ) < 0.001 )
      {
         low = 0;
         high = 1;
      }
      const double invSpan = 1.0 / ( high - low );
      for ( double& value : samples )
         value = ( value - low ) * invSpan;
      if ( !samples.empty() )
      {
         const size_t mid = samples.size() / 2;
         std::nth_element( samples.begin(), samples.begin() + ptrdiff_t( mid ), samples.end() );
         median = samples[mid]; // upper median, including for even counts
         if ( !lunar && median > 1.0e-6 )
            x = std::clamp( 0.2 / std::fma( median, median + 0.1, 1.0e-12 ), 1.0, 1.0e6 );
      }
      asinhX = std::asinh( x );
      gain = std::log10( x / asinhX );
   }

   double Forward( double value ) const
   {
      return std::asinh( ( value - low ) * ( 1.0 / ( high - low ) ) * x ) / asinhX;
   }

   double Inverse( double value ) const
   {
      return low + ( std::sinh( value * asinhX ) / x ) * ( high - low );
   }
};

struct AxisGrid
{
   int tile = 0, overlap = 0;
   float pitch = 0;
   std::vector<int> origins;

   AxisGrid( int size, int tileSize, double fraction ) : tile( tileSize )
   {
      if ( size < 1 || tile < 1 )
         throw std::invalid_argument( "FKv3 tile dimensions must be positive" );
      const float f = float( std::clamp( fraction, 0.0, 0.5 ) );
      overlap = 2 * int( std::floor( float( tile ) * f * 0.5f ) );
      const int count = size <= tile ? 1 : int( std::ceil( double( size ) / ( tile - overlap ) ) );
      pitch = float( double( size ) / count );
      for ( int k = 0; k < count; ++k )
      {
         int origin;
         if ( size < tile )
            origin = ( size - tile ) / 2; // signed division; keep the negative origin
         else if ( k == 0 )
            origin = 0;
         else if ( k == count - 1 )
            origin = size - tile;
         else
         {
            // FMUL S / FSUB S, then promote before the final subtraction.
            const float a = pitch * float( k ), b = float( tile ) - pitch;
            origin = int( std::llround( double( a ) - 0.5 * double( b ) ) );
         }
         origins.push_back( origin );
      }
   }

   std::vector<double> Window( size_t index ) const
   {
      std::vector<double> weights( size_t( tile ), 1.0 );
      constexpr double pi = 3.14159265358979323846;
      const int origin = origins.at( index );
      auto falling = [&]( int j ) { return ( std::cos( ( j + 1 ) * pi / ( overlap + 1 ) ) + 1.0 ) * 0.5; };
      if ( index > 0 )
      {
         const int left = int( std::llround( double( float( pitch * float( index ) ) ) ) ) - origin - overlap / 2;
         for ( int i = 0; i < tile; ++i )
            if ( i < left ) weights[size_t( i )] = 0;
            else if ( i < left + overlap ) weights[size_t( i )] *= 1.0 - falling( i - left );
      }
      if ( index + 1 < origins.size() )
      {
         const int right = int( std::llround( double( float( pitch * float( index + 1 ) ) ) ) ) - origin - overlap / 2;
         for ( int i = 0; i < tile; ++i )
            if ( i >= right + overlap ) weights[size_t( i )] = 0;
            else if ( i >= right ) weights[size_t( i )] *= falling( i - right );
      }
      return weights;
   }
};

// Raw model outputs remain in model space during fusion. Clip the valid
// destination rectangle before computing unsigned offsets (small images can
// have negative tile origins). Match the sink's per-tile RGB-to-mono mean
// before accumulation. add(channel,pixel,value) owns the accumulation.
template <class Add>
void BlendTile( const float* output, int channels, int imageChannels, int tileW, int tileH,
                int x0, int y0, int width, int height,
                const std::vector<double>& wx, const std::vector<double>& wy, Add add )
{
   for ( int y = std::max( 0, y0 ); y < std::min( height, y0 + tileH ); ++y )
      for ( int x = std::max( 0, x0 ); x < std::min( width, x0 + tileW ); ++x )
      {
         const int ox = x - x0, oy = y - y0;
         const size_t src = size_t( oy ) * tileW + ox, dst = size_t( y ) * width + x;
         auto weighted = [&]( int c )
         {
            return ( double( output[size_t( c ) * tileW * tileH + src] ) * wx[size_t( ox )] ) * wy[size_t( oy )];
         };
         if ( channels == 3 && imageChannels == 1 )
         {
            // MatExpr materializes R+G, then addWeighted scales that pair
            // and B by 1/3. Preserve this grouping before adding the tile.
            const double pair = weighted( 0 ) + weighted( 1 );
            constexpr double third = 1.0 / 3.0;
            add( 0, dst, pair * third + weighted( 2 ) * third );
         }
         else
            for ( int c = 0; c < channels; ++c )
               add( c, dst, weighted( c ) );
      }
}

} // namespace fkv3
} // namespace fka
