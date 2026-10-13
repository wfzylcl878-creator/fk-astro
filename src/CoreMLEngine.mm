/*
 * fk-astro — CoreML inference engine (Objective-C++ implementation)
 */
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "CoreMLEngine.h"
#include <mutex>
#include <cstring>
#include <typeinfo>
#include <cxxabi.h>

namespace fka {

static std::mutex s_mutex;

// IEEE 754 half -> float conversion (bit-level).
static float HalfToFloat( uint16_t h )
{
   uint32_t sign = uint32_t( h & 0x8000u ) << 16;
   uint32_t exp  = ( h >> 10 ) & 0x1Fu;
   uint32_t man  = h & 0x3FFu;
   uint32_t bits;
   if ( exp == 0 )
   {
      if ( man == 0 )
         bits = sign;                          // zero
      else
      {                                        // subnormal
         exp = 127 - 15 + 1;
         while ( ( man & 0x400u ) == 0 ) { man <<= 1; --exp; }
         man &= 0x3FFu;
         bits = sign | ( exp << 23 ) | ( man << 13 );
      }
   }
   else if ( exp == 0x1F )
      bits = sign | 0x7F800000u | ( man ? 1u : 0u ); // Inf/NaN
   else
      bits = sign | ( ( exp - 15 + 127 ) << 23 ) | ( man << 13 );
   float f;
   std::memcpy( &f, &bits, 4 );
   return f;
}

// Report the real type name of an in-flight C++ exception.
static std::string CurrentExceptionTypeName()
{
   const std::type_info* ti = abi::__cxa_current_exception_type();
   if ( ti == nullptr )
      return std::string();
   int status = 0;
   char* demangled = abi::__cxa_demangle( ti->name(), nullptr, nullptr, &status );
   std::string name = (status == 0 && demangled) ? demangled : ti->name();
   free( demangled );
   return name;
}

// IEEE 754 float -> half conversion (bit-level).
static uint16_t FloatToHalf( float f )
{
   uint32_t x;
   std::memcpy( &x, &f, 4 );
   uint32_t sign = ( x >> 16 ) & 0x8000u;
   int32_t  exp  = int32_t( ( x >> 23 ) & 0xFF ) - 127 + 15;
   uint32_t man  = x & 0x7FFFFF;
   if ( int32_t( ( x >> 23 ) & 0xFF ) == 0xFF ) // Inf/NaN
      return uint16_t( sign | 0x7C00u | ( man ? 1u : 0u ) );
   if ( exp >= 0x1F )
      return uint16_t( sign | 0x7C00u );        // overflow -> Inf
   if ( exp <= 0 )
      return uint16_t( sign );                   // underflow -> zero
   return uint16_t( sign | ( uint32_t( exp ) << 10 ) | ( man >> 13 ) );
}

CoreMLEngine& CoreMLEngine::Get()
{
   static CoreMLEngine instance;
   return instance;
}

void CoreMLEngine::Unload()
{
   if ( m_model != nullptr )
   {
      CFRelease( static_cast<CFTypeRef>( m_model ) );
      m_model = nullptr;
   }
   if ( m_inputName != nullptr )
   {
      CFRelease( static_cast<CFTypeRef>( m_inputName ) );
      m_inputName = nullptr;
   }
   if ( m_outputName != nullptr )
   {
      CFRelease( static_cast<CFTypeRef>( m_outputName ) );
      m_outputName = nullptr;
   }
   if ( m_paramsName != nullptr )
   {
      CFRelease( static_cast<CFTypeRef>( m_paramsName ) );
      m_paramsName = nullptr;
   }
   m_paramsSize = 0;
   m_paramsRank = 0;
   for ( int i = 0; i < 4; ++i ) m_paramsDims[i] = 1;
   m_loaded = false;
   m_loadedPath.clear();
}

// Describe all input or output ports for diagnostic messages.
static std::string DescribePorts( MLModelDescription* desc, BOOL isInput )
{
   std::string s;
   NSDictionary<NSString*, MLFeatureDescription*>* ports =
         isInput ? desc.inputDescriptionsByName : desc.outputDescriptionsByName;
   for ( NSString* name in ports )
   {
      MLFeatureDescription* fd = ports[name];
      if ( !s.empty() ) s += "; ";
      s += name.UTF8String;
      if ( fd.multiArrayConstraint != nil )
      {
         MLMultiArrayConstraint* c = fd.multiArrayConstraint;
         s += ": multiArray[";
         for ( NSNumber* d in c.shape )
            s += std::to_string( d.integerValue ) + "x";
         if ( !s.empty() && s.back() == 'x' ) s.pop_back();
         s += "]";
         if ( c.shapeConstraint != nil && c.shapeConstraint.type != MLMultiArrayShapeConstraintTypeUnspecified )
            s += " (flexible)";
      }
      else if ( fd.imageConstraint != nil )
      {
         MLImageConstraint* c = fd.imageConstraint;
         s += ": image(" + std::to_string( (int)c.pixelsWide ) + "x"
            + std::to_string( (int)c.pixelsHigh ) + ")";
      }
      else
         s += ": scalar/other";
   }
   if ( s.empty() ) s = "(none)";
   return s;
}

/*
 * Find the first usable multi-array port. Accepts fixed shapes of rank 2
 * (HxW), 3 (CHW/HWC) and 4 (1CHW/1HWC). Enumerated flexible shapes are
 * resolved to their first allowed shape.
 */
static bool GetMultiArrayPort( MLModelDescription* desc, BOOL isInput,
                               NSString** outName, TensorShape* outShape, bool* chw,
                               int* nativeRank, int* nativeDims, int* dataType )
{
   NSDictionary<NSString*, MLFeatureDescription*>* ports =
         isInput ? desc.inputDescriptionsByName : desc.outputDescriptionsByName;

   for ( NSString* name in ports )
   {
      MLFeatureDescription* fd = ports[name];
      MLMultiArrayConstraint* constraint = fd.multiArrayConstraint;
      if ( constraint == nil )
         continue;

      NSArray<NSNumber*>* shape = constraint.shape;

      // Resolve enumerated flexible shapes to the first allowed shape.
      if ( constraint.shapeConstraint != nil
           && constraint.shapeConstraint.type == MLMultiArrayShapeConstraintTypeEnumerated
           && constraint.shapeConstraint.enumeratedShapes.count > 0 )
         shape = constraint.shapeConstraint.enumeratedShapes[0];

      if ( shape.count < 2 || shape.count > 4 )
         continue;

      int d[4] = { 1, 1, 1, 1 };
      int rank = (int)shape.count;
      bool ok = true;
      for ( int i = 0; i < rank; ++i )
      {
         NSInteger v = shape[i].integerValue;
         if ( v <= 0 ) { ok = false; break; }
         d[i] = (int)v;
      }
      if ( !ok )
         continue; // free (range) dimension — unsupported

      int C, H, W;
      bool nativeCHW;
      if ( rank == 2 )
      {
         C = 1; H = d[0]; W = d[1];
         nativeCHW = true; // trivially channel-major (mono)
      }
      else if ( rank == 3 )
      {
         C = d[0]; H = d[1]; W = d[2];
         bool chFirst = ( C == 1 || C == 3 || C == 4 );
         bool chLast  = ( W == 1 || W == 3 || W == 4 );
         if ( chFirst && !chLast ) nativeCHW = true;
         else if ( chLast && !chFirst ) { nativeCHW = false; C = d[2]; H = d[0]; W = d[1]; }
         else nativeCHW = true;
      }
      else // rank == 4
      {
         if ( d[0] != 1 )
            continue; // batch dimension must be 1
         bool nchw = ( d[1] == 1 || d[1] == 3 || d[1] == 4 );
         bool nhwc = ( d[3] == 1 || d[3] == 3 || d[3] == 4 );
         if ( nchw && !nhwc )
         {
            nativeCHW = true;  C = d[1]; H = d[2]; W = d[3];
         }
         else if ( nhwc && !nchw )
         {
            nativeCHW = false; C = d[3]; H = d[1]; W = d[2];
         }
         else
         {
            nativeCHW = true;  C = d[1]; H = d[2]; W = d[3];
         }
      }

      *outName = name;
      *outShape = TensorShape{ C, H, W };
      *chw = nativeCHW;
      *nativeRank = rank;
      for ( int i = 0; i < 4; ++i ) nativeDims[i] = d[i];
      *dataType = int( constraint.dataType );
      return true;
   }
   return false;
}

// Find an optional secondary multi-array input (a parameters tensor):
// any multi-array input that is not 3-D (i.e. not the image input).
static NSString* GetParamsPort( MLModelDescription* desc, size_t* outSize,
                                int* outRank, int* outDims, int* dataType )
{
   for ( NSString* name in desc.inputDescriptionsByName )
   {
      MLFeatureDescription* fd = desc.inputDescriptionsByName[name];
      MLMultiArrayConstraint* c = fd.multiArrayConstraint;
      if ( c == nil )
         continue;

      NSArray<NSNumber*>* shape = c.shape;
      if ( shape.count < 1 || shape.count > 4 )
         continue;

      // Skip image-like ports: large 3-D/4-D tensors with spatial extent.
      if ( shape.count >= 3 )
      {
         bool spatial = false;
         for ( NSNumber* d in shape )
            if ( d.integerValue > 32 ) { spatial = true; break; }
         if ( spatial )
            continue;
      }

      size_t n = 1;
      bool valid = true;
      int rank = (int)shape.count;
      int dims[4] = { 1, 1, 1, 1 };
      for ( int i = 0; i < rank; ++i )
      {
         NSInteger v = shape[i].integerValue;
         if ( v <= 0 ) { valid = false; break; }
         dims[i] = (int)v;
         n *= size_t( v );
      }
      if ( valid && n > 0 && n <= 1024 ) // sanity: params tensors are small
      {
         *outSize = n;
         *outRank = rank;
         for ( int i = 0; i < 4; ++i ) outDims[i] = dims[i];
         *dataType = int( c.dataType );
         return name;
      }
   }
   return nil;
}

bool CoreMLEngine::BuildModel( int device )
{
   if ( m_compiledURL == nullptr )
   {
      m_lastError = "no compiled model available";
      return false;
   }

#if defined( __x86_64__ )
   /*
    * Intel Macs: the Metal (MPSGraph) backend aborts the whole process with
    * an MTLReportFailure assertion on some ops (e.g. RMSNorm), which cannot
    * be caught or fallen back from. There is no ANE either, so run on CPU.
    */
   device = DeviceCPU;
#endif

   MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
   switch ( device )
   {
   case DeviceGPU:
      cfg.computeUnits = MLComputeUnitsCPUAndGPU;
      break;
   case DeviceNE:
      cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
      break;
   case DeviceCPU:
      cfg.computeUnits = MLComputeUnitsCPUOnly;
      break;
   default:
      cfg.computeUnits = MLComputeUnitsAll;
      break;
   }

   NSURL* compiled = (__bridge NSURL*)m_compiledURL;
   NSError* error = nil;
   MLModel* model = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&error];
   if ( model == nil )
   {
      m_lastError = "MLModel load failed";
      if ( error != nil )
         m_lastError += std::string( ": " ) + error.localizedDescription.UTF8String;
      return false;
   }

   if ( m_model != nullptr )
      CFRelease( static_cast<CFTypeRef>( m_model ) );
   m_model = (void*)CFBridgingRetain( model );
   m_device = device;
   m_effectiveDevice = device;
   m_lastError.clear();
   return true;
}

bool CoreMLEngine::Load( const std::string& path, int device )
{
   std::lock_guard<std::mutex> lock( s_mutex );

   if ( m_loaded && path == m_loadedPath && device == m_requestedDevice )
      return true;

   Unload();
   m_requestedDevice = device;
   m_lastError.clear();

   // Compile once per path; reuse across compute-unit fallbacks.
   if ( m_compiledPath != path )
   {
      if ( m_compiledURL != nullptr )
      {
         CFRelease( static_cast<CFTypeRef>( m_compiledURL ) );
         m_compiledURL = nullptr;
         m_compiledPath.clear();
      }

      NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
      NSURL* url = [NSURL fileURLWithPath:nsPath];
      NSError* error = nil;
      NSURL* compiled = [MLModel compileModelAtURL:url error:&error];
      if ( compiled == nil )
      {
         m_lastError = "compileModelAtURL failed";
         if ( error != nil )
            m_lastError += std::string( ": " ) + error.localizedDescription.UTF8String;
         return false;
      }
      m_compiledURL = (void*)CFBridgingRetain( compiled );
      m_compiledPath = path;
   }

   if ( !BuildModel( device ) )
      return false;

   MLModel* model = (__bridge MLModel*)m_model;
   MLModelDescription* desc = model.modelDescription;
   NSString* inName = nil;
   NSString* outName = nil;
   TensorShape inS, outS;
   bool inCHW = true, outCHW = true;
   size_t paramsSize = 0;
   int paramsRank = 0, paramsDims[4] = { 1, 1, 1, 1 };
   int paramsDataType = int( MLMultiArrayDataTypeFloat32 );
   NSString* paramsName = GetParamsPort( desc, &paramsSize, &paramsRank, paramsDims, &paramsDataType );

   int inRank = 0,  inDims[4]  = { 1, 1, 1, 1 };
   int outRank = 0, outDims[4] = { 1, 1, 1, 1 };
   int inDataType = int( MLMultiArrayDataTypeFloat32 );
   int outDataType = int( MLMultiArrayDataTypeFloat32 );

   if ( !GetMultiArrayPort( desc, YES, &inName, &inS, &inCHW, &inRank, inDims, &inDataType ) )
   {
      m_lastError = "no usable multi-array input (fixed shape, rank 2-4, "
                    "batch dim 1). Model inputs: " + DescribePorts( desc, YES );
      return false;
   }
   if ( !GetMultiArrayPort( desc, NO, &outName, &outS, &outCHW, &outRank, outDims, &outDataType ) )
   {
      m_lastError = "no usable multi-array output. Model outputs: "
                    + DescribePorts( desc, NO );
      return false;
   }

   m_model      = (void*)CFBridgingRetain( model );
   m_inputName  = (void*)CFBridgingRetain( inName );
   m_outputName = (void*)CFBridgingRetain( outName );
   m_paramsName = paramsName ? (void*)CFBridgingRetain( paramsName ) : nullptr;
   m_paramsSize = paramsSize;
   m_paramsRank = paramsRank;
   for ( int i = 0; i < 4; ++i ) m_paramsDims[i] = paramsDims[i];
   m_inDataType = inDataType;
   m_outDataType = outDataType;
   m_paramsDataType = paramsDataType;
   m_inRank = inRank;
   for ( int i = 0; i < 4; ++i ) m_inDims[i] = inDims[i];
   m_outRank = outRank;
   for ( int i = 0; i < 4; ++i ) m_outDims[i] = outDims[i];
   m_inputCHW   = inCHW;
   m_outputCHW  = outCHW;
   m_inputShape  = inS;
   m_outputShape = outS;
   m_loadedPath  = path;
   m_loaded      = true;
   return true;
}

// (end of Load marker)

bool CoreMLEngine::Predict( const float* in, float* out, const float* params, size_t paramCount )
{
   std::lock_guard<std::mutex> lock( s_mutex );

   if ( !m_loaded )
   {
      m_lastError = "model not loaded";
      return false;
   }

   m_lastError.clear();

   for ( ;; )
   {
      std::string failure;
      @autoreleasepool
      {
      try
      {
         // N.B. In ObjC++, NSException* catch clauses must come FIRST:
         // a plain catch(...) would otherwise swallow Objective-C throws
         // and hide their diagnostic reason.
         try
         {
            if ( PredictLocked( in, out, params, paramCount ) )
               return true; // success (possibly after fallback)
         }
         catch ( NSException* e )
         {
            failure = "CoreML exception: ";
            failure += e.name.UTF8String ?: "NSException";
            if ( e.reason != nil )
            {
               failure += ": ";
               failure += e.reason.UTF8String;
            }
         }
         catch ( const std::exception& e )
         {
            failure = std::string( "C++ exception: " ) + e.what();
         }
         catch ( id generic )
         {
            failure = "CoreML: Objective-C object exception thrown";
         }
         catch ( ... )
         {
            std::string tn = CurrentExceptionTypeName();
            failure = tn.empty()
               ? "unknown C++ exception from CoreML internals"
               : "C++ exception of type " + tn + " (no message)";
         }
      }
      catch ( ... )
      {
         failure = "unexpected exception while predicting";
      }

      } // autoreleasepool

      // Try the next compute unit: All -> GPU -> CPU+ANE -> CPU.
      if ( m_effectiveDevice < DeviceCPU && BuildModel( m_effectiveDevice + 1 ) )
         continue;

      m_lastError = failure;
      if ( m_effectiveDevice != m_requestedDevice )
         m_lastError += " (after compute-unit fallback)";
      return false;
   }
}

bool CoreMLEngine::PredictLocked( const float* in, float* out, const float* params, size_t paramCount )
{
   MLModel* model     = (__bridge MLModel*)m_model;
   NSString* inName   = (__bridge NSString*)m_inputName;
   NSString* outName  = (__bridge NSString*)m_outputName;

   NSError* error = nil;

   // Build the input MLMultiArray with the model's native shape/orientation.
   NSMutableArray<NSNumber*>* shapeArr = [NSMutableArray arrayWithCapacity:m_inRank];
   for ( int i = 0; i < m_inRank; ++i )
      [shapeArr addObject:@( m_inDims[i] )];

   MLMultiArrayDataType arrayType = MLMultiArrayDataType( m_inDataType );
   if ( arrayType != MLMultiArrayDataTypeFloat16
        && arrayType != MLMultiArrayDataTypeFloat32
        && arrayType != MLMultiArrayDataTypeFloat64 )
      arrayType = MLMultiArrayDataTypeFloat32;

   MLMultiArray* array =
         [[MLMultiArray alloc] initWithShape:shapeArr
                                    dataType:arrayType
                                        error:&error];
   if ( array == nil )
   {
      m_lastError = "MLMultiArray allocation failed";
      if ( error != nil )
         m_lastError += std::string( ": " ) + error.localizedDescription.UTF8String;
      return false;
   }

   const int C = m_inputShape.c, H = m_inputShape.h, W = m_inputShape.w;
   const size_t n = m_inputShape.Size();
   if ( size_t( array.count ) != n )
   {
      m_lastError = "input multi-array element count mismatch";
      return false;
   }
   if ( arrayType == MLMultiArrayDataTypeFloat16 )
   {
      uint16_t* dst = static_cast<uint16_t*>( array.dataPointer );
      if ( m_inputCHW )
      {
         for ( size_t i = 0; i < n; ++i )
            dst[i] = FloatToHalf( in[i] );
      }
      else
      {
         for ( int c = 0; c < C; ++c )
            for ( int y = 0; y < H; ++y )
               for ( int x = 0; x < W; ++x )
                  dst[ (size_t( y )*W + x)*C + c ] =
                        FloatToHalf( in[ size_t( c )*H*W + size_t( y )*W + x ] );
      }
   }
   else
   {
      float* dst = static_cast<float*>( array.dataPointer );
      if ( m_inputCHW )
      {
         std::memcpy( dst, in, n * sizeof( float ) ); // contiguous CHW
      }
      else
      {
         for ( int c = 0; c < C; ++c )
            for ( int y = 0; y < H; ++y )
               for ( int x = 0; x < W; ++x )
                  dst[ (size_t( y )*W + x)*C + c ] = in[ size_t( c )*H*W + size_t( y )*W + x ];
      }
   }

   MLFeatureValue* featureValue = [MLFeatureValue featureValueWithMultiArray:array];
   NSMutableDictionary* features = [NSMutableDictionary dictionaryWithObject:featureValue forKey:inName];

   if ( m_paramsName != nullptr && m_paramsSize > 0 )
   {
      NSString* paramsName = (__bridge NSString*)m_paramsName;
      NSMutableArray<NSNumber*>* paramsShape = [NSMutableArray arrayWithCapacity:m_paramsRank];
      for ( int i = 0; i < m_paramsRank; ++i )
         [paramsShape addObject:@( m_paramsDims[i] )];
      MLMultiArrayDataType pType = MLMultiArrayDataType( m_paramsDataType );
      if ( pType != MLMultiArrayDataTypeFloat16 && pType != MLMultiArrayDataTypeFloat32 )
         pType = MLMultiArrayDataTypeFloat32;
      MLMultiArray* paramsArray =
            [[MLMultiArray alloc] initWithShape:paramsShape
                                      dataType:pType
                                          error:&error];
      if ( paramsArray == nil )
      {
         m_lastError = "params MLMultiArray allocation failed";
         return false;
      }
      if ( pType == MLMultiArrayDataTypeFloat16 )
      {
         uint16_t* pd = static_cast<uint16_t*>( paramsArray.dataPointer );
         for ( size_t i = 0; i < m_paramsSize; ++i )
            pd[i] = FloatToHalf( (params != nullptr && i < paramCount) ? params[i] : 0.0f );
      }
      else
      {
         float* pd = static_cast<float*>( paramsArray.dataPointer );
         for ( size_t i = 0; i < m_paramsSize; ++i )
            pd[i] = (params != nullptr && i < paramCount) ? params[i] : 0.0f;
      }
      features[paramsName] = [MLFeatureValue featureValueWithMultiArray:paramsArray];
   }

   MLDictionaryFeatureProvider* provider =
         [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
   if ( provider == nil )
   {
      m_lastError = "feature provider failed";
      if ( error != nil )
         m_lastError += std::string( ": " ) + error.localizedDescription.UTF8String;
      return false;
   }

   id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&error];
   if ( result == nil )
   {
      m_lastError = "prediction failed";
      if ( error != nil )
         m_lastError += std::string( ": " ) + error.localizedDescription.UTF8String;
      return false;
   }

   MLMultiArray* outArray = [result featureValueForName:outName].multiArrayValue;
   if ( outArray == nil )
   {
      m_lastError = "model returned no multi-array output";
      return false;
   }

   const size_t nOut = m_outputShape.Size();
   if ( size_t( outArray.count ) != nOut )
   {
      m_lastError = "output multi-array element count mismatch (declared "
                    + std::to_string( nOut ) + ", got "
                    + std::to_string( outArray.count ) + ")";
      return false;
   }

   const int oC = m_outputShape.c, oH = m_outputShape.h, oW = m_outputShape.w;
   const bool outIsF16 = ( m_outDataType == int( MLMultiArrayDataTypeFloat16 ) );

   if ( outIsF16 )
   {
      const uint16_t* src = static_cast<const uint16_t*>( outArray.dataPointer );
      if ( m_outputCHW )
      {
         for ( size_t i = 0; i < nOut; ++i )
            out[i] = HalfToFloat( src[i] );
      }
      else
      {
         for ( int c = 0; c < oC; ++c )
            for ( int y = 0; y < oH; ++y )
               for ( int x = 0; x < oW; ++x )
                  out[ size_t( c )*oH*oW + size_t( y )*oW + x ] =
                        HalfToFloat( src[ (size_t( y )*oW + x)*oC + c ] );
      }
   }
   else
   {
      const float* src = static_cast<const float*>( outArray.dataPointer );
      if ( m_outputCHW )
      {
         std::memcpy( out, src, nOut * sizeof( float ) );
      }
      else
      {
         for ( int c = 0; c < oC; ++c )
            for ( int y = 0; y < oH; ++y )
               for ( int x = 0; x < oW; ++x )
                  out[ size_t( c )*oH*oW + size_t( y )*oW + x ] = src[ (size_t( y )*oW + x)*oC + c ];
      }
   }
   return true;
}

} // namespace fka
