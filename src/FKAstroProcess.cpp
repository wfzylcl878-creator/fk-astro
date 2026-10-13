/*
 * fk-astro — FKAstroInference process implementation
 *
 * Runs a user-supplied CoreML model over the pixels of the current view,
 * using tiled inference with overlap margins and linear-ramp blending.
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
#include "CoreMLEngine.h"

#include <vector>
#include <algorithm>
#include <type_traits>
#include <chrono>

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
   enum Scheme { None = 0, FKv1 = 1 };

   Scheme scheme = None;
   double wp = 0.01;   // FKv1 working point, = 0.505 * background level

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

   float Forward( float x ) const
   {
      if ( scheme != FKv1 )
         return x;
      const double lwp = std::log( 1.0 / wp );
      const double xc = std::max( double( x ), 1.0e-12 );
      const double t = std::log( xc / wp ) / lwp;
      double z = MtfFwd( M, t );
      if ( z < 0.0 ) z = 0.0; else if ( z > 1.0 ) z = 1.0;
      return float( A * z + B );
   }

   float Inverse( float y ) const
   {
      if ( scheme != FKv1 )
         return y;
      double z = ( double( y ) - B ) / A;
      if ( z < 0.0 ) z = 0.0; else if ( z > 1.0 ) z = 1.0;
      double t = MtfInv( M, z );
      if ( t < 0.0 ) t = 0.0; else if ( t > 1.0 ) t = 1.0;
      return float( wp * std::exp( t * std::log( 1.0 / wp ) ) );
   }
};

// ---------------------------------------------------------------------------
// Background statistic (PIPELINE.md §5): bg = P35 quantile of the normalized
// image, measured once per image. Only positive samples are considered
// (zero pixels are padding/synthetic); large images are stride-sampled to
// bound the cost. Slot 5 = -log10(bg), wp = 0.505 * bg.
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
            int sy = iy + y;
            if ( reflectPad )
            {
               if ( sy < 0 ) sy = -sy;
               if ( sy >= H ) sy = 2*H - 2 - sy;
            }
            sy = std::min( std::max( sy, 0 ), H - 1 );
            float* row = dst + size_t( c )*TH*TW + size_t( y )*TW;
            const sample* srcRow = plane + size_t( sy )*W;
            for ( int x = 0; x < TW; ++x )
            {
               int sx = ix + x;
               if ( reflectPad )
               {
                  if ( sx < 0 ) sx = -sx;
                  if ( sx >= W ) sx = 2*W - 2 - sx;
               }
               sx = std::min( std::max( sx, 0 ), W - 1 );
               row[x] = normalizer->Forward( ToNormFloat( srcRow[sx] ) );
            }
         }
      }
   }
};

/*
 * Store one output tile (center-crop geometry) with linear-ramp blending:
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
            float v = normalizer->Inverse( out[ size_t( c )*outTH*outTW + size_t( oy )*outTW + ox ] );
            int dstChannel = monoModel ? pass : (replicate ? 0 : c);
            if ( replicate && C == 3 )
            {
               // mono image: average the three model output channels
               v = ( out[ size_t( 0 )*outTH*outTW + size_t( oy )*outTW + ox ]
                   + out[ size_t( 1 )*outTH*outTW + size_t( oy )*outTW + ox ]
                   + out[ size_t( 2 )*outTH*outTW + size_t( oy )*outTW + ox ] ) / 3.0f;
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


// Linear-ramp blend window for one dimension (overlap ov of n pixels).
static std::vector<float> Ramp( int n, int ov )
{
   std::vector<float> r( size_t( n ), 1.0f );
   if ( ov > 0 )
   {
      // n+2 interior points of linspace(0,1,ov+2), i.e. strictly 0..1
      std::vector<float> lin;
      for ( int i = 1; i <= ov; ++i )
         lin.push_back( float( i ) / float( ov + 1 ) );
      for ( int i = 0; i < ov && i < n; ++i )
         r[size_t( i )] = lin[size_t( i )];
      for ( int i = 0; i < ov && i < n; ++i )
         r[size_t( n - 1 - i )] = lin[size_t( ov - 1 - i )];
   }
   return r;
}

// Tile origin positions: 0..size-tile step stride, last flush to the edge.
static std::vector<int> Positions( int size, int tile, int stride )
{
   std::vector<int> pos;
   if ( size <= tile )
   {
      pos.push_back( 0 );
      return pos;
   }
   for ( int p = 0; p <= size - tile; p += stride )
      pos.push_back( p );
   if ( pos.back() != size - tile )
      pos.push_back( size - tile );
   return pos;
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
            const bool firstTile = ( t == 0 && C.verbose );
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
               pcl::Console().WriteLn( "fk-astro: stage: predict" );

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

      // 2-D blend window from per-axis linear ramps.
      const std::vector<float> ry = Ramp( outTH, outTH - C.stepY );
      const std::vector<float> rx = Ramp( outTW, outTW - C.stepX );
      win.assign( size_t( outTH )*outTW, 1.0f );
      for ( int y = 0; y < outTH; ++y )
         for ( int x = 0; x < outTW; ++x )
            win[ size_t( y )*outTW + x ] = ry[size_t( y )] * rx[size_t( x )];

      ys = Positions( H, outTH, C.stepY );
      xs = Positions( W, outTW, C.stepX );
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
            if ( ws < 1.0e-6f ) ws = 1.0f;
            float v = acc[size_t( c )][i] / ws;
            if ( v < 0.0f ) v = 0.0f; else if ( v > 1.0f ) v = 1.0f;
            d[i] = P::FloatToSample( v );
         }
      }
   }
   else if ( !C.upscale )
   {
      // In-place: same geometry as the source view.
      for ( int c = 0; c < N; ++c )
      {
         sample* d = img.PixelData( c );
         for ( size_t i = 0; i < plane; ++i )
         {
            float ws = wsum[i];
            if ( ws < 1.0e-6f ) ws = 1.0f;
            float v = acc[size_t( c )][i] / ws;
            if ( v < 0.0f ) v = 0.0f; else if ( v > 1.0f ) v = 1.0f;
            d[i] = P::FloatToSample( v );
         }
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
   //   FKv1 : per-tile log/MTF/affine normalization + 6th parameters slot
   //          (brightness statistic, computed from the background level).
   const bool fkv1 = ( int( p_normScheme ) == 1 );
   double bg = 0.0, slot5 = 0.0;
   if ( fkv1 )
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
      slot5 = -std::log10( bg );

      C.normalizer.scheme = FKNormalizer::FKv1;
      C.normalizer.wp = 0.505 * bg;

      if ( p_verbose )
         pcl::Console().WriteLn( pcl::IsoString().Format(
            "fk-astro: FKv1: bg(P35)=%.6g  slot5=-log10(bg)=%.4f  wp=0.505*bg=%.6g",
            bg, slot5, C.normalizer.wp ).c_str() );
   }

   // ---- parameters tensor payload (model contract: [1,5] or [1,6]) ----
   //   [0] sharpenStars        [1] adjustStarHalos in [0,1]
   //   [2] psfDiameter/2       [3] autoPSF flag (1/0)
   //   [4] sharpenNonstellar   [5] brightness = -log10(bg) (FKv1 only)
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
      // Slot 5: brightness statistic, computed (never a user parameter).
      if ( fkv1 && C.params.size() > 5 )
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
                                  " sharpenNonstellar", " brightness" };
         for ( size_t i = 0; i < C.params.size(); ++i )
         {
            pmsg << ( i < 6 ? names[i] : " unused" )
                 << pcl::IsoString().Format( "=%.4f", double( C.params[i] ) );
         }
         if ( p_correctOnly )
            pmsg << "  (correct only: adjustments neutralized)";
         if ( !fkv1 && C.params.size() > 5 )
            pmsg << "  (slot 5 left at 0; select FKv1 to compute it)";
      }
      else
         pmsg = "fk-astro: params: model has no parameters input";
      pcl::Console().WriteLn( pcl::String( pmsg ) );
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
