/*
 * fk-astro — FKAstroInference process implementation
 *
 * Runs a user-supplied CoreML model over the pixels of the current view,
 * using tiled inference with overlap margins and raised-cosine blending.
 */

#include <pcl/Console.h>
#include <pcl/Exception.h>
#include <pcl/View.h>
#include <pcl/StatusMonitor.h>

#include <pcl/Thread.h>
#include <pcl/MetaModule.h>
#include <pcl/ReferenceArray.h>
#include <pcl/AbstractImage.h>
#include <pcl/StandardStatus.h>
#include <pcl/AutoStatusCallbackRestorer.h>
#include <pcl/ImageWindow.h>
#include <pcl/ImageOptions.h>
#include <pcl/AutoViewLock.h>
#include <pcl/Image.h>
#include <pcl/ImageVariant.h>
#include <pcl/Utility.h>

#include "FKAstroProcess.h"
#include "FKCurves.h"
#include "CoreMLEngine.h"

#include <vector>
#include <algorithm>
#include <type_traits>
#include <chrono>
#include <cmath>
#include <limits>

namespace
{

// ------------------------------------------------------------------
// Sample -> float conversion (PixInsight float images are [0,1])
// ------------------------------------------------------------------

template <typename S>
float ToNormFloat( S v )
{
   if constexpr ( std::is_floating_point<S>::value )
      return float( v );
   else
      return float( v ) / float( std::numeric_limits<S>::max() );
}

// ---------------------------------------------------------------------------
// Per-tile normalization (PIPELINE.md §6): forward before the model, inverse
// after inference.
//
//   FKv1 forward:  t = log(clamp(x,1e-12)/wp) / log(1/wp)
//                  z = clip( MTF(t, m), 0, 1 )      with m = 0.100
//                  y = a*z + b                      a = 1.4685, b = -0.7095
//   FKv1 inverse:  z = clip( (y-b)/a, 0, 1 )
//                  t = z*m / (z*(2m-1) - (m-1))
//                  x = wp * exp( t * log(1/wp) )
//
// The working point wp = 0.505 * bg is derived once per image from the
// background statistic; the curve constants are fixed by the training
// pipeline and are not user-adjustable.
// ---------------------------------------------------------------------------
struct FKNormalizer
{
   enum Scheme { None = 0, FKv1 = 1, FKv2 = 2 };

   Scheme scheme = None;
   double wp = 0.01;   // FKv1 working point, = 0.505 * background level
   double bg = 0.01;   // image background statistic (curve selection for FKv2)
   const fka::FKCurve* curve = nullptr;   // FKv2 selected curve

   // FKv2: pick the calibration curve whose bg is closest in log space.
   void SelectCurve()
   {
      if ( scheme != FKv2 )
         return;
      double lb = std::log( std::max( bg, 1.0e-12 ) );
      double bestD = 0.0;
      for ( int i = 0; i < fka::fk_curve_count; ++i )
      {
         double d = std::fabs( std::log( fka::fk_curves[i].bg ) - lb );
         if ( curve == nullptr || d < bestD )
         {
            bestD = d;
            curve = &fka::fk_curves[i];
         }
      }
      if ( curve != nullptr )
         BuildSlopes( curve->x, curve->y, curve->n );
   }

   // Monotone PCHIP (Fritsch-Carlson) interpolation, identical to
   // scipy.interpolate.PchipInterpolator in the Python reference.
   // Two slope sets: forward (over x nodes) and inverse (over y nodes).
   double slope[512] = {};    // forward node derivatives
   double slopeI[512] = {};   // inverse node derivatives (y -> x)
   int    slopeN = 0, slopeIN = 0;

   static void BuildSlopesImpl( const double* xs, const double* ys, int n, double* out )
   {
      if ( n <= 1 )
         return;
      double d[512];
      const int nm1 = n - 1;
      for ( int i = 0; i < nm1; ++i )
      {
         double h = xs[i+1] - xs[i];
         d[i] = ( h != 0.0 ) ? ( ys[i+1] - ys[i] ) / h : 0.0;
      }
      out[0] = d[0];
      for ( int i = 1; i < nm1; ++i )
      {
         double h0 = xs[i] - xs[i-1];
         double h1 = xs[i+1] - xs[i];
         if ( d[i-1] * d[i] <= 0.0 )
            out[i] = 0.0;
         else
         {
            double w1 = 2.0*h1 + h0;
            double w2 = h1 + 2.0*h0;
            out[i] = (w1 + w2) / ( w1 / d[i-1] + w2 / d[i] );
         }
      }
      out[nm1] = d[nm1-1];
   }

   void BuildSlopes( const double* xs, const double* ys, int n )
   {
      slopeN = ( n > 0 ) ? n : 0;
      BuildSlopesImpl( xs, ys, n, slope );
      BuildSlopesImpl( ys, xs, n, slopeI );   // inverse: swap axes
      slopeIN = slopeN;
   }

   static double EvalPchip( const double* xs, const double* ys,
                            const double* m, int n, double x )
   {
      if ( n <= 1 )
         return x;
      if ( x <= xs[0] )
         return ys[0] + m[0] * ( x - xs[0] );
      if ( x >= xs[n-1] )
         return ys[n-1] + m[n-1] * ( x - xs[n-1] );
      int lo = 0, hi = n - 1;
      while ( hi - lo > 1 )
      {
         int mid = (lo + hi) >> 1;
         if ( xs[mid] <= x ) lo = mid; else hi = mid;
      }
      const double h = xs[hi] - xs[lo];
      const double t = ( x - xs[lo] ) / h;
      const double t2 = t*t, t3 = t2*t;
      const double h00 =  2*t3 - 3*t2 + 1;
      const double h10 =    t3 - 2*t2 + t;
      const double h01 = -2*t3 + 3*t2;
      const double h11 =    t3 -   t2;
      return h00*ys[lo] + h10*h*m[lo] + h01*ys[hi] + h11*h*m[hi];
   }

   double Interp( const double* xs, const double* ys, int n, double x ) const
   {
      if ( n <= 1 )
         return x;
      if ( x <= xs[0] )
         return ys[0] + slope[0] * ( x - xs[0] );
      if ( x >= xs[n-1] )
         return ys[n-1] + slope[n-1] * ( x - xs[n-1] );
      return EvalPchip( xs, ys, slope, n, x );
   }

   static double InterpLinear( const double* xs, const double* ys, int n, double x )
   {
      if ( n == 0 )
         return x;
      if ( x <= xs[0] )
      {
         if ( n > 1 )
            return ys[0] + (ys[1]-ys[0]) * (x-xs[0]) / (xs[1]-xs[0]);
         return ys[0];
      }
      if ( x >= xs[n-1] )
      {
         if ( n > 1 )
            return ys[n-1] + (ys[n-1]-ys[n-2]) * (x-xs[n-1]) / (xs[n-1]-xs[n-2]);
         return ys[n-1];
      }
      int lo = 0, hi = n - 1;
      while ( hi - lo > 1 )
      {
         int mid = (lo + hi) >> 1;
         if ( xs[mid] <= x ) lo = mid; else hi = mid;
      }
      double t = (x - xs[lo]) / (xs[hi] - xs[lo]);
      return ys[lo] + t * (ys[hi] - ys[lo]);
   }

   float Forward( float x ) const
   {
      if ( scheme == FKv2 && curve != nullptr )
         return float( Interp( curve->x, curve->y, curve->n, x ) );
      return ForwardFKv1( x );
   }

   float Inverse( float y ) const
   {
      if ( scheme == FKv2 && curve != nullptr && slopeIN == curve->n )
      {
         // Saturation semantics: model outputs beyond the calibration range
         // clamp to the curve endpoints (no extrapolation - the inverse curve
         // slope blows up near saturation and would produce wild values).
         double yv = double( y );
         const double y0 = curve->y[0], y1 = curve->y[curve->n - 1];
         if ( yv <= y0 ) return float( curve->x[0] );
         if ( yv >= y1 ) return float( curve->x[curve->n - 1] );
         return float( EvalPchip( curve->y, curve->x, slopeI, curve->n, yv ) );
      }
      return InverseFKv1( y );
   }

   static constexpr double M = 0.100;     // MTF midtone
   static constexpr double A = 1.4685;    // affine gain
   static constexpr double B = -0.7095;   // affine offset

   // PixInsight MTF: maps x == m to 0.5.
   static double MtfFwd( double m, double x )
   {
      double d = (2.0*m - 1.0)*x - m;
      return (d == 0.0) ? 0.5 : (m - 1.0)*x / d;
   }
   static double MtfInv( double m, double y )
   {
      double d = y*(2.0*m - 1.0) + 1.0 - m;
      return (d == 0.0) ? m : y*m / d;
   }

   float ForwardFKv1( float x ) const
   {
      if ( scheme == None )
         return x;
      const double lwp = std::log( 1.0 / wp );
      const double xc = std::max( double( x ), 1.0e-12 );
      const double t = std::log( xc / wp ) / lwp;
      double z = MtfFwd( M, t );
      if ( z < 0.0 ) z = 0.0; else if ( z > 1.0 ) z = 1.0;
      return float( A * z + B );
   }

   float InverseFKv1( float y ) const
   {
      if ( scheme == None )
         return y;
      double z = ( double( y ) - B ) / A;
      if ( z < 0.0 ) z = 0.0; else if ( z > 1.0 ) z = 1.0;
      double t = MtfInv( M, z );
      // Dark-end extrapolation: allow t < 0 (no clamp) so sub-background
      // model outputs map back below the background instead of clipping.
      if ( t > 1.0 ) t = 1.0;
      return float( wp * std::exp( t * std::log( 1.0 / wp ) ) );
   }
};

// ---------------------------------------------------------------------------
// Background statistic (PIPELINE.md §5): bg = P35 quantile of the normalized
// image, measured once per image. Only positive samples are considered
// (zero pixels are padding/synthetic); large images are stride-sampled to
// bound the cost. wp = 0.505 * bg, and FKv2 picks its curve from bg.
// Column 5 is log10gain, computed separately.
// ---------------------------------------------------------------------------

template <class P>
static double FKBackgroundLevelT( const pcl::GenericImage<P>& img )
{
   using sample = typename P::sample;
   const int W = img.Width(), H = img.Height(), N = img.NumberOfChannels();
   const size_t plane = size_t( W ) * size_t( H );
   const size_t total = plane * size_t( N );

   const size_t target = size_t( 2 ) << 20;   // ~2M samples
   const size_t stride = ( total > target ) ? ( total + target - 1 ) / target : 1;

   std::vector<float> samples;
   samples.reserve( std::min( total, target ) + stride );
   for ( int c = 0; c < N; ++c )
   {
      const sample* p = img.PixelData( c );
      for ( size_t i = 0; i < plane; i += stride )
      {
         float v = ToNormFloat( p[i] );
         if ( v > 0.0f )
            samples.push_back( v );
      }
   }
   if ( samples.empty() )
      return 0.0;

   const size_t k = size_t( 0.35 * double( samples.size() - 1 ) );
   std::nth_element( samples.begin(), samples.begin() + ptrdiff_t( k ), samples.end() );
   return samples[k];
}

static double FKBackgroundLevel( const pcl::ImageVariant& image )
{
   if ( image.IsFloatSample() )
      switch ( image.BitsPerSample() )
      {
      case 32: return FKBackgroundLevelT( static_cast<const pcl::Image&>( *image ) );
      case 64: return FKBackgroundLevelT( static_cast<const pcl::DImage&>( *image ) );
      }
   else
      switch ( image.BitsPerSample() )
      {
      case 8:  return FKBackgroundLevelT( static_cast<const pcl::UInt8Image&>( *image ) );
      case 16: return FKBackgroundLevelT( static_cast<const pcl::UInt16Image&>( *image ) );
      case 32: return FKBackgroundLevelT( static_cast<const pcl::UInt32Image&>( *image ) );
      }
   return 0.0;
}

// ---------------------------------------------------------------------------
// Column 5 when FKv1 or FKv2 is on: log10(x/asinh(x)).
// The ~128-px grid is nearest-neighbor because the host resampler was not
// ported. No positive samples leaves x = 1. This value does not select
// the FKv2 curve and does not feed wp.
// ---------------------------------------------------------------------------

template <class P>
static double FKLog10GainT( const pcl::GenericImage<P>& img )
{
   using sample = typename P::sample;
   const int W = img.Width(), H = img.Height(), N = img.NumberOfChannels();
   const double xOne = std::log10( 1.0 / std::asinh( 1.0 ) );
   if ( W < 1 || H < 1 || N < 1 )
      return xOne;

   int x0 = int( double( W ) * 0.05 );
   int y0 = int( double( H ) * 0.05 );
   int cropW = W - 2 * x0;
   int cropH = H - 2 * y0;
   if ( cropW < 1 || cropH < 1 )
   {
      x0 = 0;
      y0 = 0;
      cropW = W;
      cropH = H;
   }

   double high = -std::numeric_limits<double>::infinity();
   for ( int c = 0; c < N; ++c )
   {
      const sample* plane = img.PixelData( c );
      for ( int y = y0; y < y0 + cropH; ++y )
      {
         const sample* row = plane + size_t( y ) * size_t( W );
         for ( int x = x0; x < x0 + cropW; ++x )
         {
            const double v = double( ToNormFloat( row[x] ) );
            if ( std::isfinite( v ) && v > high )
               high = v;
         }
      }
   }

   const int maxSide = std::max( cropW, cropH );
   const double scale = double( maxSide ) / 128.0;
   int gridW = 1, gridH = 1;
   if ( scale > 0.0 )
   {
      gridW = std::max( 1, int( std::llround( double( cropW ) / scale ) ) );
      gridH = std::max( 1, int( std::llround( double( cropH ) / scale ) ) );
   }

   std::vector<double> positive;
   positive.reserve( size_t( gridW ) * size_t( gridH ) * size_t( N ) );
   for ( int c = 0; c < N; ++c )
   {
      const sample* plane = img.PixelData( c );
      for ( int iy = 0; iy < gridH; ++iy )
      {
         const int sy = y0 + std::min( cropH - 1,
            int( ( double( iy ) + 0.5 ) * double( cropH ) / double( gridH ) ) );
         const sample* row = plane + size_t( sy ) * size_t( W );
         for ( int ix = 0; ix < gridW; ++ix )
         {
            const int sx = x0 + std::min( cropW - 1,
               int( ( double( ix ) + 0.5 ) * double( cropW ) / double( gridW ) ) );
            const double v = double( ToNormFloat( row[sx] ) );
            if ( std::isfinite( v ) && v > 0.0 )
               positive.push_back( v );
         }
      }
   }

   double low = 0.0;
   if ( !std::isfinite( high ) )
      positive.clear();
   if ( positive.empty() )
   {
      low = 0.0;
      high = 1.0;
   }
   else
   {
      std::sort( positive.begin(), positive.end() );
      const double pos = double( positive.size() - 1 ) * 0.001;
      const size_t i0 = size_t( pos );
      const size_t i1 = std::min( i0 + 1, positive.size() - 1 );
      const double frac = pos - double( i0 );
      low = positive[i0] + frac * ( positive[i1] - positive[i0] );
   }

   if ( std::fabs( high - low ) < 0.001 )
   {
      low = 0.0;
      high = 1.0;
   }

   double x = 1.0;
   if ( !positive.empty() )
   {
      const double span = high - low;
      std::vector<double> norm( positive.size() );
      for ( size_t i = 0; i < positive.size(); ++i )
         norm[i] = ( positive[i] - low ) / span;
      const size_t mid = norm.size() / 2;
      std::nth_element( norm.begin(), norm.begin() + ptrdiff_t( mid ), norm.end() );
      const double m = norm[mid];
      if ( m > 1.0e-6 )
      {
         x = 0.2 / ( m * ( m + 0.1 ) + 1.0e-12 );
         if ( x < 1.0 )
            x = 1.0;
         else if ( x > 1.0e6 )
            x = 1.0e6;
      }
   }

   return std::log10( x / std::asinh( x ) );
}

static double FKLog10Gain( const pcl::ImageVariant& image )
{
   if ( image.IsFloatSample() )
      switch ( image.BitsPerSample() )
      {
      case 32: return FKLog10GainT( static_cast<const pcl::Image&>( *image ) );
      case 64: return FKLog10GainT( static_cast<const pcl::DImage&>( *image ) );
      }
   else
      switch ( image.BitsPerSample() )
      {
      case 8:  return FKLog10GainT( static_cast<const pcl::UInt8Image&>( *image ) );
      case 16: return FKLog10GainT( static_cast<const pcl::UInt16Image&>( *image ) );
      case 32: return FKLog10GainT( static_cast<const pcl::UInt32Image&>( *image ) );
      }
   return std::log10( 1.0 / std::asinh( 1.0 ) );
}

// Local half-float conversions for memory-lean accumulators.
static uint16_t FK_ToHalf( float f );
static float FK_ToFloat( uint16_t h );

static inline uint16_t FK_ToHalfImpl( float f )
{
   uint32_t x;
   std::memcpy( &x, &f, 4 );
   uint32_t sign = ( x >> 16 ) & 0x8000u;
   int32_t  exp  = int32_t( ( x >> 23 ) & 0xFF ) - 127 + 15;
   uint32_t man  = x & 0x7FFFFF;
   if ( int32_t( ( x >> 23 ) & 0xFF ) == 0xFF )
      return uint16_t( sign | 0x7C00u | ( man ? 1u : 0u ) );
   if ( exp >= 0x1F )
      return uint16_t( sign | 0x7C00u );
   if ( exp <= 0 )
      return uint16_t( sign );
   return uint16_t( sign | ( uint32_t( exp ) << 10 ) | ( man >> 13 ) );
}

static inline float FK_ToFloatImpl( uint16_t h )
{
   uint32_t sign = uint32_t( h & 0x8000u ) << 16;
   uint32_t exp  = ( h >> 10 ) & 0x1Fu;
   uint32_t man  = h & 0x3FFu;
   uint32_t bits;
   if ( exp == 0 )
   {
      if ( man == 0 )
         bits = sign;
      else
      {
         exp = 127 - 15 + 1;
         while ( ( man & 0x400u ) == 0 ) { man <<= 1; --exp; }
         man &= 0x3FFu;
         bits = sign | ( exp << 23 ) | ( man << 13 );
      }
   }
   else if ( exp == 0x1F )
      bits = sign | 0x7F800000u | ( man ? 1u : 0u );
   else
      bits = sign | ( ( exp - 15 + 127 ) << 23 ) | ( man << 13 );
   float f;
   std::memcpy( &f, &bits, 4 );
   return f;
}

static uint16_t FK_ToHalf( float f ) { return FK_ToHalfImpl( f ); }
static float FK_ToFloat( uint16_t h ) { return FK_ToFloatImpl( h ); }


// ------------------------------------------------------------------
// Tile helpers. Model buffers use CHW layout: buf[c*H*W + y*W + x].
// ------------------------------------------------------------------

/*
 * Type-erased per-tile image reader: reads the input window directly
 * from the view's image with edge-replicate clamping, converting to
 * normalized float32 on the fly. This avoids a full-image float copy
 * (~300 MB for a 24 MP color image) that caused heavy paging.
 */
struct FKImageReader
{
   virtual ~FKImageReader() = default;
   virtual void ReadTile( float* dst, int pass, int ix, int iy,
                          const fka::TensorShape& s,
                          bool monoModel, bool replicate ) = 0;
};

template <class P>
struct FKImageReaderT : public FKImageReader
{
   pcl::GenericImage<P>& img;
   int W, H, N;
   bool isCFA = false;
   const int (*cfaMap)[2] = nullptr;   // [y&1][x&1] -> channel (0=R,1=G,2=B)
   const FKNormalizer* normalizer = nullptr;
   bool reflectPad = true;

   FKImageReaderT( pcl::GenericImage<P>& img_ ) : img( img_ )
   {
      W = img.Width();
      H = img.Height();
      N = img.NumberOfChannels();
   }

   void ReadTile( float* dst, int pass, int ix, int iy,
                  const fka::TensorShape& s,
                  bool monoModel, bool replicate ) override
   {
      using sample = typename P::sample;
      const int C = s.c, TH = s.h, TW = s.w;

      if ( isCFA )
      {
         // Bilinear demosaic of the window from the mono Bayer plane.
         const sample* plane = img.PixelData( 0 );
         auto at = [&]( int x, int y ) -> float
         {
            if ( reflectPad )
            {
               // mirror without edge repetition: -1 -> 1, W -> W-2
               if ( x < 0 ) x = -x;
               if ( x >= W ) x = 2*W - 2 - x;
               if ( y < 0 ) y = -y;
               if ( y >= H ) y = 2*H - 2 - y;
               x = std::min( std::max( x, 0 ), W - 1 );
               y = std::min( std::max( y, 0 ), H - 1 );
            }
            else
            {
               x = std::min( std::max( x, 0 ), W - 1 );
               y = std::min( std::max( y, 0 ), H - 1 );
            }
            float v = ToNormFloat( plane[ size_t( y )*W + x ] );
            return normalizer->Forward( v );
         };

         for ( int y = 0; y < TH; ++y )
         {
            const int ay = iy + y;
            const int cy = std::min( std::max( ay, 0 ), H - 1 );
            for ( int x = 0; x < TW; ++x )
            {
               const int ax = ix + x;
               const int cx = std::min( std::max( ax, 0 ), W - 1 );
               const int sx = ax & 1, sy = ay & 1;
               const int site = cfaMap[sy][sx];

               float rgb[3] = { 0, 0, 0 };   // already normalized by at()
               if ( site == 1 ) // green site
               {
                  rgb[1] = at( cx, cy );
                  // non-green color sharing this row -> horizontal neighbors
                  const int hc = (cfaMap[sy][0] != 1) ? cfaMap[sy][0] : cfaMap[sy][1];
                  // non-green color sharing this column -> vertical neighbors
                  const int vc = (cfaMap[0][sx] != 1) ? cfaMap[0][sx] : cfaMap[1][sx];
                  rgb[hc] = (at( cx - 1, cy ) + at( cx + 1, cy )) * 0.5f;
                  rgb[vc] = (at( cx, cy - 1 ) + at( cx, cy + 1 )) * 0.5f;
               }
               else
               {
                  rgb[site] = at( cx, cy );
                  rgb[1] = (at( cx - 1, cy ) + at( cx + 1, cy )
                           + at( cx, cy - 1 ) + at( cx, cy + 1 )) * 0.25f;
                  const int diag = 2 - site; // the opposite non-green color
                  rgb[diag] = (at( cx - 1, cy - 1 ) + at( cx + 1, cy - 1 )
                              + at( cx - 1, cy + 1 ) + at( cx + 1, cy + 1 )) * 0.25f;
               }

               for ( int c = 0; c < 3 && c < C; ++c )
                  dst[ size_t( c )*TH*TW + size_t( y )*TW + x ] = rgb[c];
            }
         }
         return;
      }

      for ( int c = 0; c < C; ++c )
      {
         const int srcChannel = monoModel ? pass
                              : (replicate ? 0 : c);
         if ( srcChannel >= N )
            continue;
         const sample* plane = img.PixelData( srcChannel );
         for ( int y = 0; y < TH; ++y )
         {
            const bool zeroPad = ( normalizer->scheme == FKNormalizer::FKv2 );
            const int sy0 = iy + y;
            int sy = sy0;
            bool rowInside = true;
            if ( zeroPad )
            {
               if ( sy0 < 0 || sy0 >= H )
                  rowInside = false;                       // whole row is padding
            }
            else
            {
               if ( reflectPad )
               {
                  if ( sy < 0 ) sy = -sy;
                  if ( sy >= H ) sy = 2*H - 2 - sy;
               }
               sy = std::min( std::max( sy, 0 ), H - 1 );
            }
            float* row = dst + size_t( c )*TH*TW + size_t( y )*TW;
            if ( !rowInside )
            {
               std::fill( row, row + TW, 0.0f );           // BXT-exact: out-of-bounds stays 0
               continue;
            }
            const sample* srcRow = plane + size_t( sy )*W;
            for ( int x = 0; x < TW; ++x )
            {
               int sx = ix + x;
               if ( zeroPad )
               {
                  if ( sx < 0 || sx >= W )
                  {
                     row[x] = 0.0f;                        // padding pixel, un-normalized
                     continue;
                  }
               }
               else
               {
                  if ( reflectPad )
                  {
                     if ( sx < 0 ) sx = -sx;
                     if ( sx >= W ) sx = 2*W - 2 - sx;
                  }
                  sx = std::min( std::max( sx, 0 ), W - 1 );
               }
               row[x] = normalizer->Forward( ToNormFloat( srcRow[sx] ) );
            }
         }
      }
   }
};

/*
 * Store one output tile (center-crop geometry) with raised-cosine blending:
 * acc += piece * w ; wsum += w. Output pixel (x0+ox, y0+oy) corresponds
 * 1:1 to the tile output; blending removes seams in overlap regions.
 */
void StoreTileCrop( std::vector<std::vector<float>>& acc,
                    std::vector<float>& wsum,
                    const float* out,
                    int pass, int x0, int y0, int ph, int pw,
                    int py, int px,
                    int outTH, int outTW,
                    const std::vector<float>& win,
                    const fka::TensorShape& s, bool monoModel,
                    int W, int N, bool replicate,
                    const FKNormalizer* normalizer )
{
   const int C = s.c;
   for ( int y = 0; y < ph; ++y )
      for ( int x = 0; x < pw; ++x )
      {
         const int oy = py + y;
         const int ox = px + x;
         const float w = win[ size_t( oy )*outTW + ox ];
         const size_t di = size_t( y + y0 )*W + size_t( x + x0 );
         wsum[di] += w;
         for ( int c = 0; c < C; ++c )
         {
            if ( replicate && C == 3 && c > 0 )
               continue;   // mono image: the channel-averaged value is stored once (at c==0)
            float v = normalizer->Inverse( out[ size_t( c )*outTH*outTW + size_t( oy )*outTW + ox ] );
            int dstChannel = monoModel ? pass : (replicate ? 0 : c);
            if ( replicate && C == 3 )
            {
               // mono image: average the three model output channels in the
               // model (normalized) domain, then invert through the curve —
               // matching the Python reference (out.mean(axis=2) -> finv).
               float m3 = ( out[ size_t( 0 )*outTH*outTW + size_t( oy )*outTW + ox ]
                          + out[ size_t( 1 )*outTH*outTW + size_t( oy )*outTW + ox ]
                          + out[ size_t( 2 )*outTH*outTW + size_t( oy )*outTW + ox ] ) / 3.0f;
               v = normalizer->Inverse( m3 );
               dstChannel = 0;
            }
            acc[size_t( dstChannel )][di] += v * w;
         }
      }
}

/*
 * Store one output tile in scaled (upscale) mode: each destination pixel
 * is owned by the tile whose input-space center region contains it.
 */
void StoreTileScaled( std::vector<std::vector<float>>& dst,
                      const float* out,
                      int pass, int x0, int y0, int x1, int y1,
                      int ix, int iy,
                      const fka::TensorShape& s, bool monoModel,
                      float sx, float sy, int outW, int outH,
                      const FKNormalizer* normalizer )
{
   const int C = s.c, TH = s.h, TW = s.w;
   const int XoBase = int( ix*sx + 0.5f );
   const int YoBase = int( iy*sy + 0.5f );
   const float fx0 = float( x0 ), fx1 = float( x1 );
   const float fy0 = float( y0 ), fy1 = float( y1 );
   for ( int oy = 0; oy < TH; ++oy )
   {
      const int Yo = YoBase + oy;
      if ( Yo < 0 || Yo >= outH ) continue;
      const float Yi = float( Yo ) / sy;
      if ( Yi < fy0 || Yi >= fy1 ) continue;
      for ( int ox = 0; ox < TW; ++ox )
      {
         const int Xo = XoBase + ox;
         if ( Xo < 0 || Xo >= outW ) continue;
         const float Xi = float( Xo ) / sx;
         if ( Xi < fx0 || Xi >= fx1 ) continue;
         for ( int c = 0; c < C; ++c )
         {
            const int dstChannel = monoModel ? pass : c;
            float v = out[ size_t( c )*TH*TW + size_t( oy )*TW + ox ];
            if ( v < 0.0f ) v = 0.0f; else if ( v > 1.0f ) v = 1.0f;
            dst[size_t( dstChannel )][ size_t( Yo )*outW + Xo ] = v;
         }
      }
   }
}


// Raised cosine of length ov. Samples fall from ~1 to ~0; the ~0 end sits
// on the outer border so two tiles whose origins differ by tile-ov sum to 1.
static std::vector<float> Ramp( int n, int ov )
{
   std::vector<float> r( size_t( std::max( n, 0 ) ), 1.0f );
   if ( ov <= 0 || n <= 0 )
      return r;
   constexpr double kPi = 3.14159265358979323846;
   const int len = std::min( ov, n );
   for ( int i = 0; i < len; ++i )
   {
      const int hostIndex = ov - i;
      const double w = ( std::cos( hostIndex * kPi / double( ov + 1 ) ) + 1.0 ) * 0.5;
      r[size_t( i )] = float( w );
      r[size_t( n - 1 - i )] = float( w );
   }
   return r;
}

// Overlap implied by the middle step of an equal-pitch origin list.
static int MidOverlap( const std::vector<int>& pos, int tile )
{
   if ( pos.size() < 2 || tile <= 1 )
      return 0;
   const size_t i = pos.size() / 2;
   const size_t i0 = ( i == 0 ) ? 0 : i - 1;
   if ( i0 + 1 >= pos.size() )
      return 0;
   int step = pos[i0 + 1] - pos[i0];
   if ( step < 1 )
      step = 1;
   int ov = tile - step;
   if ( ov < 0 )
      ov = 0;
   if ( ov >= tile )
      ov = tile - 1;
   return ov;
}

// Tile origin positions - PixInsight BXT-exact grid (reverse engineered from
// RCAstroSuite-pxm.dylib FUN_000f089c and verified against its runtime tile
// coordinates): n = ceil(size / (tile*0.8)); sx = size/n (float);
// pos(k) = round(sx*k - (tile-sx)/2) for k = 1..n-2, first tile 0, last tile
// size-tile. With size=6248, tile=512 this yields
// 0,330,720,1111,1501,...,5406,5736 - exactly the BXT runtime sequence.
static std::vector<int> Positions( int size, int tile )
{
   std::vector<int> pos;
   if ( size <= tile )
   {
      pos.push_back( std::max( 0, ( size - tile ) / 2 ) );
      return pos;
   }
   const int n = int( std::ceil( double( size ) / ( double( tile ) * 0.8 ) ) );
   const double sx = double( size ) / n;
   pos.push_back( 0 );
   for ( int k = 1; k <= n - 2; ++k )
      pos.push_back( int( std::lround( sx * k - ( double( tile ) - sx ) / 2.0 ) ) );
   pos.push_back( size - tile );
   // Enforce bounds and strict monotonicity.
   for ( int& p : pos )
      p = std::min( std::max( p, 0 ), size - tile );
   std::vector<int> uniq;
   for ( int p : pos )
      if ( uniq.empty() || p > uniq.back() )
         uniq.push_back( p );
   if ( uniq.back() != size - tile )
      uniq.push_back( size - tile );
   return uniq;
}

// ------------------------------------------------------------------
// Type-dispatched pipeline
// ------------------------------------------------------------------

struct RunConfig
{
   fka::CoreMLEngine* engine  = nullptr;
   int                W = 0, H = 0, N = 0;
   fka::TensorShape   inS, outS;
   bool               monoModel = false;  // model input has 1 channel
   bool               replicate = false;  // model wants C>1, image is mono
   // center-crop geometry (out <= in): margins around the output tile
   int                marginX = 0, marginY = 0;
   int                stepX = 0, stepY = 0;      // stride in image domain
   int                outTileW = 0, outTileH = 0;
   // CFA (raw Bayer) input: per-tile demosaic -> RGB, re-mosaic on write.
   bool               isCFA = false;
   int                cfaMap[2][2] = { { 0, 1 }, { 1, 2 } }; // channel per site (x&1, y&1)
   // scaled geometry (out > in)
   bool               upscale = false;
   float              sx = 1.0f, sy = 1.0f;
   int                outW = 0, outH = 0;
   double             tileOverlap = 0.2;
   bool               verbose = true;
   FKNormalizer       normalizer;
   bool               reflectPad = true;  // catalog: reflectionPad
   std::vector<float> params;             // parameters tensor payload
};

// ------------------------------------------------------------------
// Worker thread: processes a contiguous range of tiles.
// ------------------------------------------------------------------

struct FKAstroThreadData : public pcl::AbstractImage::ThreadData
{
   FKAstroThreadData( const pcl::AbstractImage& image, pcl::size_type N )
      : pcl::AbstractImage::ThreadData( image, N )
   {
   }

   const RunConfig* C = nullptr;
   FKImageReader* reader = nullptr;
   const FKNormalizer* normalizer = nullptr;
   // center-crop mode
   std::vector<std::vector<float>>* acc = nullptr;
   std::vector<float>* wsum = nullptr;
   const std::vector<float>* win = nullptr;
   // upscale mode
   std::vector<std::vector<float>>* dst = nullptr;
   // tile grid
   const std::vector<int>* ys = nullptr;
   const std::vector<int>* xs = nullptr;
   int passes = 1;
};

class FKAstroTileThread : public pcl::Thread
{
public:

   using size_type = pcl::size_type;

   FKAstroTileThread( FKAstroThreadData& data, size_type begin, size_type end )
      : m_data( data ), m_begin( begin ), m_end( end )
   {
   }

   void Run() override
   {
      INIT_THREAD_MONITOR()

      try
      {
         const RunConfig& C = *m_data.C;
         const FKNormalizer* normalizer = m_data.normalizer;
         const int W = C.W, H = C.H, N = C.N;
         const auto& ys = *m_data.ys;
         const auto& xs = *m_data.xs;
         const size_type nx = size_type( xs.size() );
         const size_type perPass = size_type( ys.size() ) * nx;

         std::vector<float> inBuf( C.inS.Size() );
         std::vector<float> outBuf( C.outS.Size() );
         const int outTH = C.outTileH, outTW = C.outTileW;

         for ( size_type t = m_begin; t < m_end; ++t )
         {
            const bool firstTile = ( t - m_begin < 2 && C.verbose );
            const int pass = int( t / perPass );
            const size_type r = t % perPass;
            const int y = ys[ int( r / nx ) ];
            const int x = xs[ int( r % nx ) ];

            const auto tt = std::chrono::steady_clock::now();
            const bool timingTile = ( t - m_begin < 6 );

            if ( firstTile )
               pcl::Console().WriteLn( "fk-astro: stage: fill" );

            const int ix = x - C.marginX;
            const int iy = y - C.marginY;

            m_data.reader->ReadTile( inBuf.data(), pass, ix, iy,
                                     C.inS, C.monoModel, C.replicate );

            if ( firstTile )
            {
               pcl::Console().WriteLn( "fk-astro: stage: predict" );
               if ( C.verbose )
               {
                  double s0 = 0, s1 = 0, mn = 1e30, mx = -1e30;
                  for ( size_t i = 0; i < inBuf.size() && i < 3072*3; ++i )
                  {
                     s0 += inBuf[i];
                     if ( inBuf[i] < mn ) mn = inBuf[i];
                     if ( inBuf[i] > mx ) mx = inBuf[i];
                  }
                  const size_t nn = std::min( inBuf.size(), size_t( 3072*3 ) );
                  s1 = s0 / double( nn );
                  pcl::Console().WriteLn( pcl::IsoString().Format(
                     "fk-astro: tile0 input (normalized): mean=%.5f min=%.5f max=%.5f",
                     s1, mn, mx ).c_str() );
               }
            }

            if ( !C.engine->Predict( inBuf.data(), outBuf.data(),
                                     C.params.data(), C.params.size() ) )
            {
               m_error = ( "fk-astro: prediction failed: "
                           + C.engine->LastError() ).c_str();
               return;
            }

            if ( firstTile )
               pcl::Console().WriteLn( "fk-astro: stage: store" );

            if ( !C.upscale )
            {
               const int ph = std::min( outTH, H - y );
               const int pw = std::min( outTW, W - x );
               const int py = ( y + outTH > H ) ? outTH - ph : 0;
               const int px = ( x + outTW > W ) ? outTW - pw : 0;

               StoreTileCrop( *m_data.acc, *m_data.wsum, outBuf.data(), pass,
                              x, y, ph, pw, py, px, outTH, outTW, *m_data.win,
                              C.outS, C.monoModel, W, N, C.replicate, normalizer );
            }
            else
            {
               StoreTileScaled( *m_data.dst, outBuf.data(), pass,
                                x, y, std::min( x + C.stepX, W ),
                                std::min( y + C.stepY, H ),
                                ix, iy, C.outS, C.monoModel,
                                C.sx, C.sy, C.outW, C.outH, normalizer );
            }

            if ( firstTile )
               pcl::Console().WriteLn( "fk-astro: stage: tile 1 done" );

            if ( timingTile )
            {
               double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - tt ).count();
               pcl::Console().WriteLn( pcl::IsoString().Format(
                  "fk-astro: [thread %d] tile %d: %.1f ms",
                  int( m_begin > 0 ), int( t - m_begin ), ms ).c_str() );
            }

            UPDATE_THREAD_MONITOR( 1 )
         }
      }
      catch ( const std::exception& e )
      {
         m_error = pcl::String( ( std::string( "fk-astro: thread exception: " )
                                  + e.what() ).c_str() );
      }
      catch ( ... )
      {
         m_error = "fk-astro: unknown thread exception";
      }
   }

   pcl::String m_error;

private:

   FKAstroThreadData& m_data;
   size_type          m_begin;
   size_type          m_end;
};

// ------------------------------------------------------------------
// Type-dispatched pipeline
// ------------------------------------------------------------------

template <class P>
void RunPipeline( pcl::GenericImage<P>& img, const RunConfig& C )
{
   using sample = typename P::sample;

   const int W = C.W, H = C.H, N = C.N;
   const size_t plane = size_t( W )*H;

   // Per-tile reader: no full-image float snapshot (memory + paging).
   FKImageReaderT<P> reader( img );
   reader.isCFA = C.isCFA;
   reader.cfaMap = C.cfaMap;
   reader.normalizer = &C.normalizer;
   reader.reflectPad = C.reflectPad;

   const int passes = C.monoModel ? N : 1;

   // Shared output accumulators / destination.
   std::vector<std::vector<float>> acc;
   std::vector<float> wsum;
   std::vector<float> win;
   std::vector<std::vector<float>> dst;
   std::vector<int> ys, xs;
   int outTH = C.outTileH, outTW = C.outTileW;

   if ( !C.upscale )
   {
      acc.assign( size_t( N ), std::vector<float>( plane, 0.0f ) );
      wsum.assign( plane, 0.0f );

      // Equal-pitch origins first. The cosine length is the middle gap,
      // tile minus that step. Measuring ys before Positions() saw an
      // empty list and collapsed the blend to one pixel.
      ys = Positions( H, outTH );
      xs = Positions( W, outTW );
      const int rampY = MidOverlap( ys, outTH );
      const int rampX = MidOverlap( xs, outTW );
      const std::vector<float> ry = Ramp( outTH, rampY );
      const std::vector<float> rx = Ramp( outTW, rampX );
      win.assign( size_t( outTH )*outTW, 1.0f );
      for ( int y = 0; y < outTH; ++y )
         for ( int x = 0; x < outTW; ++x )
            win[ size_t( y )*outTW + x ] = ry[size_t( y )] * rx[size_t( x )];
   }
   else
   {
      const size_t outPlane = size_t( C.outW )*C.outH;
      dst.assign( size_t( N ), std::vector<float>( outPlane, 0.0f ) );

      const int TW = C.inS.w, TH = C.inS.h;
      const int m = C.marginX;
      const int stepX = std::max( 1, C.stepX );
      const int stepY = std::max( 1, C.stepY );
      // In upscale mode the tile grid walks the input image directly.
      for ( int y0 = 0; y0 < H; y0 += stepY )
         ys.push_back( y0 );
      for ( int x0 = 0; x0 < W; x0 += stepX )
         xs.push_back( x0 );
      (void)TW; (void)TH; (void)m;
   }

   const size_t total = size_t( ys.size() ) * size_t( xs.size() ) * size_t( passes );

   // Attach a StandardStatus callback to the image's status monitor: the
   // periodic callback dispatch performed by RunThreads while waiting for
   // the worker threads is what generates live console progress, keeps the
   // GUI responsive and supports cancellation. The restorer puts back the
   // previous callback when this pipeline returns.
   pcl::StandardStatus stdStatus;
   pcl::AutoStatusCallbackRestorer restoreStatusCallback( img.Status() );
   img.SetStatusCallback( &stdStatus );

   // 2. Run tiles on worker threads via the canonical RunThreads
   //    mechanism: while waiting, the image's status callbacks are
   //    dispatched periodically, which keeps the GUI responsive, shows
   //    live progress, and supports cancellation.
   FKAstroThreadData data( img, total );
   data.normalizer = &C.normalizer;

   /*
    * ThreadData's constructor only copies the image's monitor: it never
    * initializes it. An uninitialized monitor is never registered with
    * MonitorDispatcherThread, so m_needsUpdate is never set and Updated()
    * callbacks never fire (no live percentage). Initialize() called here
    * (on the root thread) registers the monitor with the dispatcher.
    */
   data.status.Initialize( pcl::String( "fk-astro: applying CoreML model" ),
                           pcl::size_type( total ) );

   data.C = &C;
   data.reader = &reader;
   data.acc = &acc;
   data.wsum = &wsum;
   data.win = &win;
   data.dst = &dst;
   data.ys = &ys;
   data.xs = &xs;
   data.passes = passes;

   pcl::ReferenceArray<FKAstroTileThread> threads;
   const size_t half = ( total + 1 ) / 2;
   threads.Add( new FKAstroTileThread( data, 0, half ) );
   threads.Add( new FKAstroTileThread( data, half, total ) );

   // Required by UPDATE_THREAD_MONITOR(): workers take the synchronized
   // count/mutex path only when numThreads > 1 (RunThreads() would set
   // this, but we run our own event-pumping wait loop).
   data.numThreads = threads.Length();

   if ( C.verbose )
      pcl::Console().WriteLn( "" ); // start progress on a fresh console line

   /*
    * N.B. ExecuteOn runs on the ROOT thread (Apply button dispatch), so a
    * plain blocking wait would freeze the GUI (Thread::Wait parks the
    * thread in pthread_cond_wait without running the Qt event loop).
    * Instead of AbstractImage::RunThreads(), we use our own wait loop:
    * wake every 10 ms, transfer the worker progress count into the image's
    * status monitor (firing StandardStatus console updates on this thread)
    * and pump application events with MetaModule::ProcessEvents().
    */
   for ( FKAstroTileThread& thread : threads )
      thread.Start( pcl::ThreadPriority::Normal );

   {
      bool done0 = false, done1 = false;
      size_t lastCount = 0;
      while ( !( done0 && done1 ) )
      {
         if ( !done0 && threads[0].Wait( 10 ) )
            done0 = true;
         if ( !done1 && threads[1].Wait( 10 ) )
            done1 = true;

         if ( data.mutex.TryLock() )
         {
            try
            {
               if ( data.total > 0 && data.count > lastCount )
               {
                  data.status += data.count - lastCount;
                  lastCount = data.count;
               }
               data.mutex.Unlock();
            }
            catch ( ... )
            {
               data.mutex.Unlock();
               for ( FKAstroTileThread& thread : threads )
                  thread.Abort();
               for ( FKAstroTileThread& thread : threads )
                  thread.Wait();
               threads.Destroy();
               throw pcl::ProcessAborted();
            }
         }

         pcl::Module->ProcessEvents( /*excludeUserInputEvents*/ true );
      }
   }

   for ( FKAstroTileThread& thread : threads )
      if ( !thread.m_error.IsEmpty() )
      {
         threads.Destroy();
         throw pcl::Error( thread.m_error );
      }
   threads.Destroy();

   if ( C.verbose )
   {
      // Main-thread diagnostics: accumulate stats before writing back.
      double amin = 1e30, amax = -1e30, asum = 0; size_t anz = 0;
      for ( int c = 0; c < int( acc.size() ); ++c )
         for ( size_t i = 0; i < acc[c].size(); i += 7 )
         {
            double v = acc[c][i];
            if ( v < amin ) amin = v;
            if ( v > amax ) amax = v;
            asum += v; ++anz;
         }
      double wmin = 1e30, wmax = -1e30, wsum_v = 0; size_t wnz = 0;
      for ( size_t i = 0; i < wsum.size(); i += 7 )
      {
         double v = wsum[i];
         if ( v < wmin ) wmin = v;
         if ( v > wmax ) wmax = v;
         wsum_v += v; ++wnz;
      }
      pcl::Console().WriteLn( pcl::IsoString().Format(
         "fk-astro: diag: acc(min=%.5g max=%.5g mean=%.5g n=%zu)  w(min=%.5g max=%.5g mean=%.5g n=%zu)",
         amin, amax, ( anz ? asum/anz : 0 ), anz,
         wmin, wmax, ( wnz ? wsum_v/wnz : 0 ), wnz ).c_str() );
   }

   // 3. Write results back.
   if ( !C.upscale && C.isCFA )
   {
      // Re-mosaic: keep the raw Bayer structure, sample the channel that
      // matches each pixel's CFA site.
      sample* d = img.PixelData( 0 );
      for ( int y = 0; y < H; ++y )
      {
         const int sy = y & 1;
         for ( int x = 0; x < W; ++x )
         {
            const int c = C.cfaMap[sy][x & 1];
            const size_t i = size_t( y )*W + x;
            float ws = wsum[i];
            // Cosine borders are legitimately << 1e-6. Only a true zero
            // means no tile wrote this pixel.
            if ( !( ws > 0.0f ) ) ws = 1.0f;
            float v = acc[size_t( c )][i] / ws;
            if ( v < 0.0f ) v = 0.0f; else if ( v > 1.0f ) v = 1.0f;
            if constexpr ( std::is_floating_point<sample>::value )
               d[i] = sample( v );
            else
               d[i] = sample( double( v ) * double( P::HighestSampleValue() ) + 0.5 );
         }
      }
   }
   else if ( !C.upscale )
   {
      // In-place: same geometry as the source view.
      for ( int c = 0; c < N; ++c )
      {
         sample* d = img.PixelData( c );
         const size_t i0 = size_t( C.H / 2 ) * size_t( C.W ) + size_t( C.W / 2 );
         const bool dbg = ( c == 0 && C.verbose && i0 < plane );
         for ( size_t i = 0; i < plane; ++i )
         {
            float ws = wsum[i];
            if ( !( ws > 0.0f ) ) ws = 1.0f;
            float v = acc[size_t( c )][i] / ws;
            if ( v < 0.0f ) v = 0.0f; else if ( v > 1.0f ) v = 1.0f;
            if constexpr ( std::is_floating_point<sample>::value )
               d[i] = sample( v );
            else
               d[i] = sample( double( v ) * double( P::HighestSampleValue() ) + 0.5 );
            if ( dbg && i == i0 )
               pcl::Console().WriteLn( pcl::IsoString().Format(
                  "fk-astro: diag: pixel %zu: acc=%.6g ws=%.6g v=%.6g d=%.6g",
                  i0, acc[0][i0], wsum[i0], v, double( d[i0] ) ).c_str() );
         }
      }
      if ( C.verbose )
      {
         sample* d = img.PixelData( 0 );
         double mn = 1e30, mx = -1e30, sm = 0; size_t k = 0;
         for ( size_t i = 0; i < plane; i += 7 )
         {
            double v = double( d[i] );
            if ( v < mn ) mn = v;
            if ( v > mx ) mx = v;
            sm += v; ++k;
         }
         pcl::Console().WriteLn( pcl::IsoString().Format(
            "fk-astro: diag: after writeback: raw sample d: min=%.6g max=%.6g mean=%.6g (first 3: %g %g %g)",
            mn, mx, sm / std::max<size_t>( k, 1 ),
            double( d[0] ), double( d[1] ), double( d[2] ) ).c_str() );
      }
   }
   else
   {
      // Scaled output: create a new float32 image window.
      const size_t outPlane = size_t( C.outW )*C.outH;
      pcl::ImageWindow window( C.outW, C.outH, N,
                               /*bitsPerSample*/ 32,
                               /*floatSample*/ true,
                               /*color*/ N >= 3,
                               /*initialProcessing*/ true,
                               "FKAstroInference_output" );
      pcl::View outView = window.MainView();
      pcl::AutoViewLock outLock( outView );
      pcl::ImageVariant outImage = outView.Image();
      pcl::Image& outImg = static_cast<pcl::Image&>( *outImage );
      for ( int c = 0; c < N; ++c )
      {
         float* d = outImg.PixelData( c );
         const float* s = dst[size_t( c )].data();
         for ( size_t i = 0; i < outPlane; ++i )
            d[i] = s[i];
      }
      window.Show();
   }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Parameter singletons
// ---------------------------------------------------------------------------

FKAstroInferenceProcess* TheFKAstroInferenceProcess = nullptr;

FKModelPathParameter*    TheFKModelPathParameter    = nullptr;
FKSharpenStarsParameter* TheFKSharpenStarsParameter = nullptr;
FKAdjustStarHalosParameter* TheFKAdjustStarHalosParameter = nullptr;
FKSharpenNonstellarParameter* TheFKSharpenNonstellarParameter = nullptr;
FKAutoPSFParameter* TheFKAutoPSFParameter = nullptr;
FKPSFDiameterParameter* TheFKPSFDiameterParameter = nullptr;
FKCorrectOnlyParameter* TheFKCorrectOnlyParameter = nullptr;
FKLunarPlanetaryParameter* TheFKLunarPlanetaryParameter = nullptr;
FKNormSchemeParameter* TheFKNormSchemeParameter = nullptr;
FKReflectPadParameter* TheFKReflectPadParameter = nullptr;
FKTileOverlapParameter*  TheFKTileOverlapParameter  = nullptr;
FKDeviceParameter*       TheFKDeviceParameter       = nullptr;
FKVerboseParameter*      TheFKVerboseParameter      = nullptr;

// ---------------------------------------------------------------------------
// FKAstroInferenceProcess
// ---------------------------------------------------------------------------

FKAstroInferenceProcess::FKAstroInferenceProcess()
   : pcl::MetaProcess()
{
   TheFKAstroInferenceProcess = this;
   TheFKModelPathParameter    = nullptr;
   new FKModelPathParameter( this );
   new FKSharpenStarsParameter( this );
   new FKAdjustStarHalosParameter( this );
   new FKSharpenNonstellarParameter( this );
   new FKAutoPSFParameter( this );
   new FKPSFDiameterParameter( this );
   new FKCorrectOnlyParameter( this );
   new FKNormSchemeParameter( this );
   new FKReflectPadParameter( this );
   new FKTileOverlapParameter( this );
   new FKDeviceParameter( this );
   new FKVerboseParameter( this );
   // Appended so the FKv2 parameter ids keep their previous positions.
   new FKLunarPlanetaryParameter( this );
}

pcl::IsoString FKAstroInferenceProcess::Id() const
{
   return "FKAstroInference";
}

pcl::IsoString FKAstroInferenceProcess::Aliases() const
{
   return "CoreMLInference,fkAstroInference";
}

pcl::IsoString FKAstroInferenceProcess::Categories() const
{
   return "Preprocessing,FKAstro";
}

pcl::IsoString FKAstroInferenceProcess::IconImageSVG() const
{
   return
      "<svg xmlns='http://www.w3.org/2000/svg' width='128' height='128'>"
      "<defs>"
      "<linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
      "<stop offset='0' stop-color='#0b3d5c'/><stop offset='1' stop-color='#17a2b8'/>"
      "</linearGradient>"
      "</defs>"
      "<rect width='128' height='128' rx='18' fill='url(#g)'/>"
      "<circle cx='64' cy='56' r='20' fill='#e9f6fb' opacity='0.9'/>"
      "<ellipse cx='64' cy='102' rx='38' ry='10' fill='#e9f6fb' opacity='0.45'/>"
      "<path d='M28 96 L52 74 M100 96 L76 74' stroke='#ffd166' stroke-width='6' stroke-linecap='round'/>"
      "</svg>";
}

pcl::ProcessImplementation* FKAstroInferenceProcess::Create() const
{
   return new FKAstroInferenceInstance( this );
}

pcl::ProcessImplementation* FKAstroInferenceProcess::Clone( const pcl::ProcessImplementation& p ) const
{
   FKAstroInferenceInstance* instance = new FKAstroInferenceInstance( this );
   instance->Assign( p );
   return instance;
}

// ---------------------------------------------------------------------------
// FKAstroInferenceInstance
// ---------------------------------------------------------------------------

FKAstroInferenceInstance::FKAstroInferenceInstance( const FKAstroInferenceProcess* p )
   : pcl::ProcessImplementation( p )
{
}

void FKAstroInferenceInstance::Assign( const pcl::ProcessImplementation& x )
{
   const FKAstroInferenceInstance& p = dynamic_cast<const FKAstroInferenceInstance&>( x );
   p_modelPath    = p.p_modelPath;
   p_sharpenStars      = p.p_sharpenStars;
   p_adjustStarHalos   = p.p_adjustStarHalos;
   p_sharpenNonstellar = p.p_sharpenNonstellar;
   p_autoPSF      = p.p_autoPSF;
   p_psfDiameter  = p.p_psfDiameter;
   p_tileOverlap  = p.p_tileOverlap;
   p_correctOnly  = p.p_correctOnly;
   p_lunarPlanetary = p.p_lunarPlanetary;
   p_normScheme   = p.p_normScheme;
   p_reflectPad   = p.p_reflectPad;
   p_device       = p.p_device;
   p_verbose      = p.p_verbose;
}

bool FKAstroInferenceInstance::Validate( pcl::String& info )
{
   if ( p_modelPath.IsEmpty() )
   {
      info = "No CoreML model file has been specified.";
      return false;
   }
   return true;
}

bool FKAstroInferenceInstance::CanExecuteOn( const pcl::View& view, pcl::String& whyNot ) const
{
   if ( !view.IsMainView() )
   {
      whyNot = "fk-astro can only be executed on main views.";
      return false;
   }
   if ( view.Image().IsComplexSample() )
   {
      whyNot = "fk-astro cannot be executed on complex images.";
      return false;
   }
   return true;
}

bool FKAstroInferenceInstance::ExecuteOn( pcl::View& view )
{
   if ( p_modelPath.IsEmpty() )
      throw pcl::Error( "fk-astro: no CoreML model specified." );

   pcl::AutoViewLock lock( view );

   if ( p_verbose )
      pcl::Console().WriteLn( pcl::IsoString().Format(
         "fk-astro: ExecuteOn on %s thread",
         pcl::Thread::IsRootThread() ? "ROOT" : "worker" ).c_str() );

   const auto t0 = std::chrono::steady_clock::now();

   pcl::ImageVariant image = view.Image();
   if ( image.Width() < 1 || image.Height() < 1 )
      throw pcl::Error( "fk-astro: the target view is empty." );
   if ( image.IsComplexSample() )
      throw pcl::Error( "fk-astro cannot be executed on complex images." );

   fka::CoreMLEngine& engine = fka::CoreMLEngine::Get();
   if ( !engine.Load( pcl::IsoString( p_modelPath ).c_str(), int( p_device ) ) )
      throw pcl::Error( ( "fk-astro: cannot load model: "
                          + engine.LastError() ).c_str() );

   const fka::TensorShape inS  = engine.InputShape();
   const fka::TensorShape outS = engine.OutputShape();
   if ( !inS.IsValid() || !outS.IsValid() )
      throw pcl::Error( "fk-astro: the model must expose usable multi-array ports." );
   if ( outS.c != inS.c )
      throw pcl::Error( pcl::IsoString().Format(
         "fk-astro: model has %d input channel(s) but %d output channel(s); "
         "these must match.", inS.c, outS.c ) );

   RunConfig C;
   C.engine = &engine;
   C.W = image.Width();
   C.H = image.Height();
   C.N = image.NumberOfChannels();
   C.inS = inS;
   C.outS = outS;
   C.verbose = p_verbose;
   C.reflectPad = p_reflectPad;

   // ---- normalization scheme (PIPELINE.md) ----
   //   None : pixels pass through, 5 user parameter slots only.
   //   FKv1 : log/MTF/affine curve. FKv2 : calibrated PCHIP curve.
   //          Both write column 5 as log10gain. wp and FKv2 selection use P35.
   const int normScheme = int( p_normScheme );
   const bool needs6 = ( normScheme >= 1 );   // FKv1 and FKv2 both use 6 params
   double bg = 0.0, slot5 = 0.0;
   if ( needs6 )
   {
      if ( !engine.HasParamsInput() || engine.ParamsSize() < 6 )
         throw pcl::Error( pcl::IsoString().Format(
            "fk-astro: FKv1 normalization requires a model with a 6-element "
            "parameters input; this model declares %d element(s). Select "
            "'None' for 5-parameter models.",
            engine.HasParamsInput() ? int( engine.ParamsSize() ) : 0 ) );

      bg = FKBackgroundLevel( image );
      if ( bg <= 0.0 )
         throw pcl::Error( "fk-astro: FKv1: the image has no positive pixels; "
                           "cannot derive the background statistic." );
      bg = std::min( std::max( bg, 1.0e-12 ), 1.0 );
      slot5 = FKLog10Gain( image );

      C.normalizer.scheme = (normScheme == 2) ? FKNormalizer::FKv2
                                             : FKNormalizer::FKv1;
      C.normalizer.wp = 0.505 * bg;
      C.normalizer.bg = bg;
      C.normalizer.SelectCurve();

      if ( p_verbose )
         pcl::Console().WriteLn( pcl::IsoString().Format(
            "fk-astro: FKv%c: log10gain=%.6g  bg(P35)=%.6g  wp=0.505*bg=%.6g%s",
            normScheme == 2 ? '2' : '1',
            slot5, bg, C.normalizer.wp,
            (normScheme == 2 && C.normalizer.curve)
               ? pcl::IsoString().Format( "  curve bg=%.5g (%d nodes)",
                    C.normalizer.curve->bg, C.normalizer.curve->n ).c_str()
               : "" ).c_str() );
   }

   // ---- parameters tensor payload (model contract: [1,5] or [1,6]) ----
   //   [0] sharpenStars        [1] adjustStarHalos in [0,1]
   //   [2] psfDiameter/2       [3] autoPSF flag (1/0)
   //   [4] sharpenNonstellar   [5] log10gain (FKv1 and FKv2)
   if ( engine.HasParamsInput() )
   {
      C.params.assign( engine.ParamsSize(), 0.0f );
      if ( p_correctOnly )
      {
         // Correct Only: all adjustments are neutralized at runtime.
         //   [0] sharpenStars    = 0
         //   [1] adjustStarHalos = 0
         //   [2] psfDiameter/2   = 0
         //   [3] autoPSF flag    = 1 (automatic PSF still applies)
         //   [4] sharpenNonstellar = 0
         if ( C.params.size() > 0 ) C.params[0] = 0.0f;
         if ( C.params.size() > 1 ) C.params[1] = 0.0f;
         if ( C.params.size() > 2 ) C.params[2] = 0.0f;
         if ( C.params.size() > 3 ) C.params[3] = 1.0f;
         if ( C.params.size() > 4 ) C.params[4] = 0.0f;
      }
      else
      {
         if ( C.params.size() > 0 )
            C.params[0] = float( p_sharpenStars );
         if ( C.params.size() > 1 )
            C.params[1] = float( p_adjustStarHalos );  // slider value passed as-is
         if ( C.params.size() > 2 )
            C.params[2] = float( p_autoPSF ? 0.0 : p_psfDiameter / 2.0 );
         if ( C.params.size() > 3 )
            C.params[3] = float( p_autoPSF ? 1.0 : 0.0 );
         if ( C.params.size() > 4 )
            C.params[4] = float( p_sharpenNonstellar );
      }
      // lunarPlanetary is a mode flag and is not a column.
      if ( needs6 && C.params.size() > 5 )
         C.params[5] = float( slot5 );
   }

   // ---- verbose: report the exact parameters tensor payload ----
   if ( p_verbose )
   {
      pcl::IsoString pmsg;
      if ( engine.HasParamsInput() )
      {
         pmsg << "fk-astro: params["
              << pcl::IsoString().Format( "%d", int( C.params.size() ) ) << "]:";
         const char* names[6] = { " sharpenStars", " adjustStarHalos",
                                  " psfDiameter/2", " autoPSF",
                                  " sharpenNonstellar", " log10gain" };
         for ( size_t i = 0; i < C.params.size(); ++i )
         {
            pmsg << ( i < 6 ? names[i] : " unused" )
                 << pcl::IsoString().Format( "=%.4f", double( C.params[i] ) );
         }
         if ( p_correctOnly )
            pmsg << "  (correct only: adjustments neutralized)";
         if ( !needs6 && C.params.size() > 5 )
            pmsg << "  (slot 5 left at 0; select FKv1 to compute it)";
      }
      else
         pmsg = "fk-astro: params: model has no parameters input";
      pcl::Console().WriteLn( pcl::String( pmsg ) );
      pcl::Console().WriteLn( pcl::IsoString().Format(
         "fk-astro: lunarPlanetary=%s (mode flag, not a parameters column)",
         p_lunarPlanetary ? "on" : "off" ).c_str() );
   }

   // ---- channel mapping ----
   // ---- CFA (raw Bayer) detection ----
   const int cfaType = view.Window().FileInfo().cfaType;
   C.isCFA = ( cfaType >= pcl::CFAType::BGGR && cfaType <= pcl::CFAType::RGGB );
   if ( C.isCFA )
   {
      if ( C.upscale )
         throw pcl::Error( "fk-astro: CFA images are not supported with "
                           "upscale models (debayer first)." );
      if ( inS.c != 3 )
         throw pcl::Error( "fk-astro: CFA images require a 3-channel model "
                           "(or debayer the image first)." );
      switch ( cfaType )
      {
      case pcl::CFAType::GRBG: C.cfaMap[0][0]=1; C.cfaMap[0][1]=0; C.cfaMap[1][0]=2; C.cfaMap[1][1]=1; break;
      case pcl::CFAType::GBRG: C.cfaMap[0][0]=1; C.cfaMap[0][1]=2; C.cfaMap[1][0]=0; C.cfaMap[1][1]=1; break;
      case pcl::CFAType::BGGR: C.cfaMap[0][0]=2; C.cfaMap[0][1]=1; C.cfaMap[1][0]=1; C.cfaMap[1][1]=0; break;
      default: /* RGGB */      C.cfaMap[0][0]=0; C.cfaMap[0][1]=1; C.cfaMap[1][0]=1; C.cfaMap[1][1]=2; break;
      }
      C.N = 3;              // logical RGB processing; write-back re-mosaics
      C.monoModel = false;
      C.replicate = false;
      if ( p_verbose )
         pcl::Console().WriteLn( pcl::IsoString().Format(
            "fk-astro: CFA input detected (pattern %d): per-tile demosaic -> "
            "RGB -> re-mosaic", cfaType ).c_str() );
   }
   else
   {
      if ( cfaType == pcl::CFAType::CYGM )
         throw pcl::Error( "fk-astro: CYGM CFA images are not supported." );
      C.monoModel = ( inS.c == 1 && C.N > 1 );     // mono model: per-channel passes
      C.replicate = ( inS.c > 1 && C.N == 1 );     // color model on mono image
   }
   if ( !C.monoModel && !C.replicate && !C.isCFA && inS.c != C.N )
      throw pcl::Error( pcl::IsoString().Format(
         "fk-astro: the model expects %d input channel(s); the image has %d channel(s).",
         inS.c, C.N ) );

   // ---- geometry ----
   const int outTileH = outS.h, outTileW = outS.w;
   C.outTileW = outTileW;
   C.outTileH = outTileH;
   const double frac = std::min( std::max( p_tileOverlap, 0.0 ), 0.5 );

   if ( outTileH <= inS.h && outTileW <= inS.w )
   {
      // Center-crop contract: input window = output tile + margins.
      C.upscale = false;
      C.marginX = ( inS.w - outTileW ) / 2;
      C.marginY = ( inS.h - outTileH ) / 2;
      C.stepX = std::max( 1, int( std::lround( outTileW * ( 1.0 - frac ) ) ) );
      C.stepY = std::max( 1, int( std::lround( outTileH * ( 1.0 - frac ) ) ) );
      C.outW = C.W;
      C.outH = C.H;
   }
   else
   {
      // Upscale contract: output is a scaled image in a new window.
      C.upscale = true;
      C.sx = float( outTileW ) / float( inS.w );
      C.sy = float( outTileH ) / float( inS.h );
      C.marginX = 0;
      C.stepX = std::max( 1, int( std::lround( inS.w * ( 1.0 - frac ) ) ) );
      C.stepY = std::max( 1, int( std::lround( inS.h * ( 1.0 - frac ) ) ) );
      C.outW = int( float( C.W ) * C.sx + 0.5f );
      C.outH = int( float( C.H ) * C.sy + 0.5f );
      if ( C.outW < 1 || C.outH < 1 )
         throw pcl::Error( "fk-astro: invalid output image geometry." );
   }

   if ( p_verbose )
   {
      pcl::IsoString summary;
      if ( C.upscale )
         summary.Format( "fk-astro: in %dx%dx%d -> out %dx%dx%d (x%.3g) | image %dx%d -> %dx%d | stride %dx%d",
                         inS.c, inS.h, inS.w, outS.c, outS.h, outS.w, double( C.sx ),
                         C.H, C.W, C.outH, C.outW, C.stepY, C.stepX );
      else
         summary.Format( "fk-astro: window %dx%dx%d -> center %dx%d | margin %d | stride %dx%d | image %dx%dx%d",
                         inS.c, inS.h, inS.w, outTileW, outTileH,
                         C.marginX, C.stepY, C.stepX, C.N, C.H, C.W );
      pcl::Console().WriteLn( pcl::String( summary ) );
   }

   try
   {
      if ( image.IsFloatSample() )
         switch ( image.BitsPerSample() )
         {
         case 32: RunPipeline( static_cast<pcl::Image&>( *image ), C ); break;
         case 64: RunPipeline( static_cast<pcl::DImage&>( *image ), C ); break;
         default: throw pcl::Error( "fk-astro: unsupported floating point sample format." );
         }
      else
         switch ( image.BitsPerSample() )
         {
         case 8:  RunPipeline( static_cast<pcl::UInt8Image&>( *image ), C ); break;
         case 16: RunPipeline( static_cast<pcl::UInt16Image&>( *image ), C ); break;
         case 32: RunPipeline( static_cast<pcl::UInt32Image&>( *image ), C ); break;
         default: throw pcl::Error( "fk-astro: unsupported integer sample format." );
         }
   }
   catch ( pcl::Error& )
   {
      throw;
   }
   catch ( const std::exception& e )
   {
      throw pcl::Error( ( std::string( "fk-astro: std exception in pipeline: " )
                          + e.what() ).c_str() );
   }
   catch ( ... )
   {
      throw pcl::Error( "fk-astro: unknown exception in pipeline "
                        "(outside CoreML prediction)" );
   }

   if ( p_verbose )
   {
      double sec = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0 ).count();
      pcl::Console().WriteLn( pcl::IsoString().Format(
         "fk-astro: done in %.1f s", sec ).c_str() );
   }

   if ( p_verbose && engine.EffectiveDevice() != int( p_device ) )
   {
      const char* devName = "?";
      switch ( engine.EffectiveDevice() )
      {
      case fka::DeviceAuto: devName = "Auto"; break;
      case fka::DeviceGPU:  devName = "GPU"; break;
      case fka::DeviceNE:   devName = "CPU+ANE"; break;
      case fka::DeviceCPU:  devName = "CPU"; break;
      }
      pcl::Console().WriteLn( pcl::IsoString().Format(
         "fk-astro: note: compute device fell back to %s", devName ).c_str() );
   }

   return true;
}

void* FKAstroInferenceInstance::LockParameter( const pcl::MetaParameter* p, pcl::size_type /*tableRow*/ )
{
   if ( p == TheFKModelPathParameter )    return &p_modelPath;   if ( p == TheFKSharpenStarsParameter ) return &p_sharpenStars;
   if ( p == TheFKAdjustStarHalosParameter ) return &p_adjustStarHalos;
   if ( p == TheFKSharpenNonstellarParameter ) return &p_sharpenNonstellar;
   if ( p == TheFKAutoPSFParameter ) return &p_autoPSF;
   if ( p == TheFKPSFDiameterParameter ) return &p_psfDiameter;
   if ( p == TheFKCorrectOnlyParameter ) return &p_correctOnly;
   if ( p == TheFKLunarPlanetaryParameter ) return &p_lunarPlanetary;
   if ( p == TheFKNormSchemeParameter ) return &p_normScheme;
   if ( p == TheFKReflectPadParameter ) return &p_reflectPad;
   if ( p == TheFKTileOverlapParameter )  return &p_tileOverlap;
   if ( p == TheFKDeviceParameter )       return &p_device;
   if ( p == TheFKVerboseParameter )      return &p_verbose;
   return pcl::ProcessImplementation::LockParameter( p, 0 );
}

bool FKAstroInferenceInstance::AllocateParameter( pcl::size_type sizeOrLength, const pcl::MetaParameter* p, pcl::size_type tableRow )
{
   if ( p == TheFKModelPathParameter )
   {
      p_modelPath.Clear();
      if ( sizeOrLength > 0 )
         p_modelPath.SetLength( sizeOrLength );
      return true;
   }
   return pcl::ProcessImplementation::AllocateParameter( sizeOrLength, p, tableRow );
}

pcl::size_type FKAstroInferenceInstance::ParameterLength( const pcl::MetaParameter* p, pcl::size_type tableRow ) const
{
   if ( p == TheFKModelPathParameter )
      return p_modelPath.Length();
   return pcl::ProcessImplementation::ParameterLength( p, tableRow );
}
