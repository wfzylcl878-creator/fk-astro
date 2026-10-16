#include "FKv3Pipeline.h"
#include "TensorLayout.h"
#include "CoreMLEngine.h"
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include "module_reader.inc" // generated from the actual process source

using namespace fka::fkv3;

static void Check( bool value, const char* message )
{
   if ( !value ) throw std::runtime_error( message );
}
static void Near( double a, double b, double tolerance, const char* message )
{
   Check( std::isfinite( a ) && std::isfinite( b ) && std::fabs( a-b ) <= tolerance, message );
}
template <class At>
static ArcsinhNormalizer Fit( int w, int h, int n, bool lunar, At at )
{
   ArcsinhNormalizer state;
   state.Fit( w, h, n, lunar, at );
   return state;
}

static void ReaderRoundtrip( int width, int height, int channels )
{
   pcl::GenericImage<TestFloat> image( width, height, channels );
   for ( int c = 0; c < channels; ++c )
      for ( int y = 0; y < height; ++y )
         for ( int x = 0; x < width; ++x )
            image.pixels[size_t( c )*width*height + size_t( y )*width+x] =
               float( 0.003 + 0.01*c + 0.3*x/width + 0.1*y/height );
   FKNormalizer norm;
   norm.scheme = FKNormalizer::FKv3;
   norm.fkv3 = FitFKv3T( image, false );
   FKImageReaderT<TestFloat> reader( image );
   reader.normalizer = &norm;
   AxisGrid gx( width, 512, 0.2 ), gy( height, 512, 0.2 );
   std::vector<std::vector<double>> acc( size_t( channels ), std::vector<double>( size_t( width )*height, 0 ) );
   std::vector<float> input( 3*1024*1024 ), output( 3*512*512 );
   for ( size_t j = 0; j < gy.origins.size(); ++j )
      for ( size_t i = 0; i < gx.origins.size(); ++i )
      {
         const int x = gx.origins[i], y = gy.origins[j];
         reader.ReadTile( input.data(), 0, x-256, y-256, fka::TensorShape{3,1024,1024}, false, channels == 1 );
         // A known synthetic model: return the central 512, unchanged.
         for ( int c = 0; c < 3; ++c )
            for ( int py = 0; py < 512; ++py )
               for ( int px = 0; px < 512; ++px )
                  output[size_t( c )*512*512+size_t( py )*512+px] = input[size_t( c )*1024*1024+size_t( py+256 )*1024+px+256];
         BlendTile( output.data(), 3, channels, 512, 512, x, y, width, height, gx.Window( i ), gy.Window( j ),
            [&]( int c, size_t pixel, double value ) { acc[size_t( c )][pixel] += value; } );
      }
   const auto original = image.pixels;
   WriteFKv3T( image, norm.fkv3, acc );
   for ( int c = 0; c < channels; ++c )
      for ( size_t i = 0; i < size_t( width )*height; ++i )
         Near( image.pixels[size_t( c )*width*height+i], original[size_t( c )*width*height+i], 8e-8,
               "center-copy host roundtrip (including seams and outer borders)" );
}

int main()
{
   int passed = 0;
   auto run = [&]( const char* name, const std::function<void()>& body )
   {
      body(); ++passed; std::cout << "PASS " << name << '\n';
   };
   try
   {
      run( "empty-positive fallback", []
      {
         const auto n = Fit( 31, 17, 3, false, []( int, int, int ) { return 0.0; } );
         Check( n.positiveSamples == 0 && n.low == 0 && n.high == 1 && n.x == 1, "black image must remain valid" );
         Near( n.gain, 0.05483996855669083, 1e-15, "fallback gain is not zero" );
      } );
      run( "upper median and fixed white point", []
      {
         const auto n = Fit( 2, 1, 1, false, []( int, int, int x ) { return x ? 0.9 : 0.1; } );
         Check( n.positiveSamples == 8192 && n.high == 1, "upsampling and wp=1" );
         Near( n.low, 0.1, 1e-15, "positive percentile" );
         Near( n.median, 0.8/0.9, 1e-15, "even counts must use upper median" );
      } );
      run( "floor nearest coordinates and pooled channels", []
      {
         std::vector<int> sx;
         const auto n = Fit( 181, 139, 3, false, [&]( int c, int y, int x )
         {
            if ( c == 0 && y == 6 ) sx.push_back( x );
            return c == 0 ? 0.0 : c == 1 ? 0.004 : 0.02;
         } );
         Check( n.cropX == 9 && n.cropY == 6 && n.gridW == 128 && n.gridH == 100, "crop/grid dimensions" );
         Check( sx.size() == 128 && sx[2] == 11, "no half-pixel offset" );
         Check( n.positiveSamples == 25600, "pool positive channel samples, without luminance averaging" );
         Near( n.low, 0.004, 1e-15, "low quantile" );
         Near( n.median, 0.016/0.996, 1e-15, "upper pooled median" );
      } );
      run( "interpolated low quantile", []
      {
         const auto n = Fit( 181, 1, 1, false, []( int, int, int x ) { return (x+1)*0.001; } );
         // 128 samples: q=.127, s[0]=.010, s[1]=.011.
         Near( n.low, 0.010127, 1e-15, "0.1% quantile must interpolate adjacent samples" );
      } );
      run( "near-white span fallback", []
      {
         const auto n = Fit( 1, 1, 1, false, []( int, int, int ) { return 0.9996; } );
         Check( n.low == 0 && n.high == 1, "reset only when high-low is too small" );
      } );
      run( "independent offline reference fixture", []
      {
         // Golden values from the separately recovered Python reference,
         // on this deterministic float32 gradient; not official runtime data.
         const auto n = Fit(198,153,3,false,[](int c,int y,int x)
            { return double(float(.003+.00001*x+.00002*y+c*.0003)); });
         Check(n.positiveSamples==38016 && n.gridW==128 && n.gridH==99,"reference sample grid");
         Near(n.low,0.0033701500471215694,1e-17,"reference low");
         Near(n.median,0.002417998745413611,1e-17,"reference median");
         Near(n.x,807.6024676707292,1e-11,"reference x");
         Near(n.gain,2.0387167509049915,1e-13,"reference gain");
         Check(float(n.gain)==2.0387167930603027f,"reference float32 slot");
      } );
      run( "mode overrides and six columns", []
      {
         Settings s;
         s.stars=.56; s.halos=-.1; s.diameter=3.2; s.nonstellar=.37; s.lunar=true;
         auto p=s.Parameters( 1.25 );
         Check( p[0]==0 && p[1]==0 && p[2]==1.6f && p[3]==0 && p[4]==.37f && p[5]==1.25f, "lunar disabledIf and sinkScale" );
         s.correctOnly=true; p=s.Parameters( 1.25 );
         Check( !s.Effective().lunar && p[2]==0 && p[3]==1 && p[4]==0 && p[5]==1.25f, "Correct Only pins override lunar but keep gain" );
         const auto n=Fit( 4,4,3,true,[]( int, int y, int x ) { return .002+.001*(x+y); } );
         Check( n.x==1, "lunar keeps x=1" );
      } );
      run( "analytic inverse without input clamping", []
      {
         const auto n=Fit( 7,5,3,false,[]( int c, int y, int x ) { return .002+.001*(x+y+c); } );
         for ( double p : {-0.1,0.0,0.003,0.1,1.0,1.2} ) Near( n.Inverse(n.Forward(p)),p,2e-14,"inverse roundtrip" );
         Check( n.Forward(0)<0 && n.x>1, "do not clamp input below low" );
      } );
      run( "float64 and integer sample precision", []
      {
         pcl::GenericImage<TestDouble> image( 1,1,1 ); image.pixels[0]=0.100000000002;
         FKNormalizer norm; norm.scheme=FKNormalizer::FKv3; norm.fkv3=FitFKv3T(image,false);
         FKImageReaderT<TestDouble> reader(image); reader.normalizer=&norm;
         float value=1; reader.ReadTile(&value,0,0,0,fka::TensorShape{1,1,1},false,false);
         Near(value,0,1e-12,"do not cast double pixels to float before normalization");
         Near(SampleToDouble(uint16_t(12345)),12345.0/65535.0,1e-16,"integer conversion in double");
         pcl::GenericImage<TestUInt16> integer(1,1,1); integer.pixels[0]=12345;
         const auto state=FitFKv3T(integer,false);
         WriteFKv3T(integer,state,std::vector<std::vector<double>>(1,std::vector<double>(1,0)));
         Check(integer.pixels[0]==12345,"integer output must be scaled and rounded");
         Check(SampleFromDouble<uint8_t>(.5/255)==0 && SampleFromDouble<uint8_t>(1.5/255)==2
               && SampleFromDouble<uint8_t>(2.5/255)==2,"integer half ties round to even");
      } );
      run( "padding is model zero, valid zero is transformed", []
      {
         pcl::GenericImage<TestFloat> image(2,1,1); image.pixels={0.0f,0.005f};
         FKNormalizer norm; norm.scheme=FKNormalizer::FKv3; norm.fkv3=FitFKv3T(image,false);
         FKImageReaderT<TestFloat> reader(image); reader.normalizer=&norm; reader.reflectPad=true;
         float values[4]={}; reader.ReadTile(values,0,-1,0,fka::TensorShape{1,1,4},false,false);
         Check(values[0]==0 && values[1]<0 && values[2]==0 && values[3]==0,"padding must bypass Forward even when reflectPad was saved on");
      } );
      run( "tile count, mixed precision and negative origins", []
      {
         Check(AxisGrid(6150,512,.2).origins.size()==15,"overlap .2 tile count");
         Check(AxisGrid(6150,512,.1).origins.size()==14,"overlap .1 tile count");
         Check(AxisGrid(1025,512,.2).origins==std::vector<int>({0,256,513}),"promote before final origin subtraction");
         Check(AxisGrid(317,512,.2).origins==std::vector<int>({-97}),"small-image centering");
         Check(AxisGrid(512,512,.2).origins==std::vector<int>({0}),"exact tile size");
      } );
      run( "seam windows, zero overlap and outer borders", []
      {
         for ( int size : {317,512,513,1025,6150,6248} )
            for ( double overlap : {0.0,0.1,0.2,0.5} )
            {
               AxisGrid g(size,512,overlap); std::vector<double> sum(size_t(size),0);
               for(size_t k=0;k<g.origins.size();++k)
               {
                  const auto w=g.Window(k);
                  for(int i=0;i<512;++i) if(g.origins[k]+i>=0 && g.origins[k]+i<size)
                     sum[size_t(g.origins[k]+i)]+=w[size_t(i)];
               }
               double expectedPeak = 1;
               // Saved static reference: transition bands overlap at .5.
               // Preserve that host behavior instead of renormalizing it.
               if ( overlap == .5 )
               {
                  if ( size == 513 ) expectedPeak=1.0045559170634588;
                  if ( size == 1025 ) expectedPeak=1.0006270795355663;
                  if ( size == 6150 ) expectedPeak=1.0000012550459407;
                  if ( size == 6248 ) expectedPeak=1.0000003571197926;
               }
               for(double w:sum) Check(w>=1-1e-12 && w<=expectedPeak+1e-12,"axis coverage");
               Near(*std::max_element(sum.begin(),sum.end()),expectedPeak,1e-12,"reference peak weight sum");
               Near(sum.front(),1,1e-15,"left border not tapered"); Near(sum.back(),1,1e-15,"right border not tapered");
            }
      } );
      run( "model-domain sum before inverse, no weight division", []
      {
         double acc=0; const float a=.4f,b=.6f;
         BlendTile(&a,1,1,1,1,0,0,1,1,{.2},{1},[&](int,size_t,double v){acc+=v;});
         BlendTile(&b,1,1,1,1,0,0,1,1,{.3},{1},[&](int,size_t,double v){acc+=v;});
         Near(acc,double(a)*.2+double(b)*.3,1e-16,"raw weighted sum");
         auto n=Fit(4,4,1,false,[](int,int y,int x){return .003+.001*(x+y);});
         Check(std::fabs(n.Inverse(acc)-(.2*n.Inverse(a)+.3*n.Inverse(b)))>1e-5,"per-tile inverse is not equivalent");
      } );
      run( "unequal mono outputs reduced per tile before inverse", []
      {
         // Independent scalar fixture from the recovered sink expression:
         // addWeighted(R+G, 1/3, B, 1/3, 0), then add to a mono image.
         const float outputs[4][3] = {
            {.07979197800159454f, .23279088735580444f, .10100142657756805f},
            {.27797359228134155f, .6356844305992126f, .36483219265937805f},
            {.37018096446990967f, .2095070332288742f, .26697781682014465f},
            {.9366545677185059f, .6480354070663452f, .6091309785842896f}
         };
         const double wx[4] = {.22789827565154686, .7721017243484531, .22789827565154686, .7721017243484531};
         const double wy[4] = {.28938796360210717, .28938796360210717, .7106120363978928, .7106120363978928};
         std::vector<std::vector<double>> acc( 1, std::vector<double>( 1, 0 ) );
         int writes = 0;
         for ( int t = 0; t < 4; ++t )
            BlendTile( outputs[t], 3, 1, 1, 1, 0, 0, 1, 1, {wx[t]}, {wy[t]},
               [&]( int c, size_t pixel, double value )
               {
                  Check( c == 0 && pixel == 0, "mono sink must receive only channel zero" );
                  ++writes; acc[0][0] += value;
               } );
         Check( writes == 4, "reduce each tile before its single mono accumulation" );
         Near( acc[0][0], .5512419161912594, 0, "recovered sink grouping and double accumulation" );
         auto n = Fit(4,4,1,false,[](int,int y,int x){return .003+.001*(x+y);});
         pcl::GenericImage<TestDouble> image(1,1,1);
         WriteFKv3T(image,n,acc);
         Near(image.pixels[0],n.Inverse(.5512419161912594),1e-15,"inverse after mono fusion");
      } );
      run( "raw CFA plane uses mono replication without interpolation", []
      {
         // Distinct RGGB sites must stay at their original coordinates.
         pcl::GenericImage<TestFloat> image(2,2,1);
         image.pixels = {.02f,.10f,.20f,.80f};
         FKNormalizer norm; norm.scheme = FKNormalizer::FKv3; norm.fkv3 = FitFKv3T(image,false);
         FKImageReaderT<TestFloat> reader(image); reader.normalizer = &norm;
         std::vector<float> input(3*4*4);
         reader.ReadTile(input.data(),0,-1,-1,fka::TensorShape{3,4,4},false,true);
         for ( int c = 0; c < 3; ++c )
            for ( int y = 0; y < 4; ++y )
               for ( int x = 0; x < 4; ++x )
               {
                  const bool inside = x >= 1 && x <= 2 && y >= 1 && y <= 2;
                  const float expected = inside ? float(norm.fkv3.Forward(double(image.pixels[size_t(y-1)*2+x-1]))) : 0;
                  Near(input[size_t(c)*16+size_t(y)*4+x],expected,0,"same raw site in all channels, model-zero padding");
               }
      } );
      run( "small RGB center-copy pipeline", [] { ReaderRoundtrip(317,241,3); } );
      run( "small mono center-copy pipeline", [] { ReaderRoundtrip(317,241,1); } );
      run( "multiple tiles center-copy pipeline", [] { ReaderRoundtrip(901,673,3); } );
      run( "strided image and parameters layouts", []
      {
         Check(fka::ImageOffset({10000,300,30,2},false,2,4,5)==1354,"strided NHWC batch");
         Check(fka::ImageOffset({1000,100,2},true,2,4,5)==2410,"strided CHW");
         Check(fka::ImageOffset({17,2},true,0,4,5)==78,"strided mono rank two");
         Check(fka::LinearOffset(5,{1,6},{24,3})==15,"strided six-column tensor");
         double data[20]={};
         fka::WriteArrayValue(data,fka::MLMultiArrayDataTypeFloat64,15,.25f);
         Check(data[15]==.25 && data[14]==0 && data[16]==0,"write float64 at element stride");
         Check(fka::ReadArrayValue(data,fka::MLMultiArrayDataTypeFloat64,15)==.25f,"read runtime float64 dtype");
         uint16_t half[3]={}; fka::WriteArrayValue(half,fka::MLMultiArrayDataTypeFloat16,2,.5f);
         Check(fka::ReadArrayValue(half,fka::MLMultiArrayDataTypeFloat16,2)==.5f,"half conversion at element stride");
      } );
      std::cout << passed << " offline cases passed; no CoreML/PixInsight runtime executed.\n";
      return 0;
   }
   catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
