#pragma once

/*
 * fk-astro — FKAstroInference process definition
 */

#include <pcl/ProcessImplementation.h>
#include "FKAstroInterface.h"
#include <pcl/MetaProcess.h>
#include <pcl/MetaParameter.h>
#include <pcl/String.h>


class FKAstroInterface;
extern FKAstroInterface* TheFKAstroInterface;
class FKAstroInferenceProcess;
extern FKAstroInferenceProcess* TheFKAstroInferenceProcess;

// forward declarations
class FKModelPathParameter;
class FKSharpenStarsParameter;
class FKAdjustStarHalosParameter;
class FKSharpenNonstellarParameter;
class FKTileOverlapParameter;
class FKDeviceParameter;
class FKVerboseParameter;

extern FKModelPathParameter*    TheFKModelPathParameter;
extern FKSharpenStarsParameter* TheFKSharpenStarsParameter;
extern FKAdjustStarHalosParameter* TheFKAdjustStarHalosParameter;
extern FKSharpenNonstellarParameter* TheFKSharpenNonstellarParameter;
class FKAutoPSFParameter;
class FKPSFDiameterParameter;
extern FKAutoPSFParameter* TheFKAutoPSFParameter;
extern FKPSFDiameterParameter* TheFKPSFDiameterParameter;
class FKCorrectOnlyParameter;
extern FKCorrectOnlyParameter* TheFKCorrectOnlyParameter;
class FKLunarPlanetaryParameter;
extern FKLunarPlanetaryParameter* TheFKLunarPlanetaryParameter;
class FKNormSchemeParameter;
class FKReflectPadParameter;
extern FKNormSchemeParameter* TheFKNormSchemeParameter;
extern FKReflectPadParameter* TheFKReflectPadParameter;
extern FKTileOverlapParameter*  TheFKTileOverlapParameter;
extern FKDeviceParameter*       TheFKDeviceParameter;
extern FKVerboseParameter*      TheFKVerboseParameter;

// ---------------------------------------------------------------------------
// Process parameter metaclasses
// ---------------------------------------------------------------------------

class FKModelPathParameter : public pcl::MetaString
{
public:
   FKModelPathParameter( pcl::MetaProcess* P ) : pcl::MetaString( P )
   {
      TheFKModelPathParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "modelPath";
   }
};





class FKSharpenStarsParameter : public pcl::MetaFloat
{
public:
   FKSharpenStarsParameter( pcl::MetaProcess* P ) : pcl::MetaFloat( P )
   {
      TheFKSharpenStarsParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "sharpenStars";
   }
};

class FKAdjustStarHalosParameter : public pcl::MetaFloat
{
public:
   FKAdjustStarHalosParameter( pcl::MetaProcess* P ) : pcl::MetaFloat( P )
   {
      TheFKAdjustStarHalosParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "adjustStarHalos";
   }
};

class FKSharpenNonstellarParameter : public pcl::MetaFloat
{
public:
   FKSharpenNonstellarParameter( pcl::MetaProcess* P ) : pcl::MetaFloat( P )
   {
      TheFKSharpenNonstellarParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "sharpenNonstellar";
   }
};

class FKAutoPSFParameter : public pcl::MetaBoolean
{
public:
   FKAutoPSFParameter( pcl::MetaProcess* P ) : pcl::MetaBoolean( P )
   {
      TheFKAutoPSFParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "autoPSF";
   }
};

class FKPSFDiameterParameter : public pcl::MetaFloat
{
public:
   FKPSFDiameterParameter( pcl::MetaProcess* P ) : pcl::MetaFloat( P )
   {
      TheFKPSFDiameterParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "psfDiameter";
   }
};

class FKTileOverlapParameter : public pcl::MetaFloat
{
public:
   FKTileOverlapParameter( pcl::MetaProcess* P ) : pcl::MetaFloat( P )
   {
      TheFKTileOverlapParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "tileOverlap";
   }
};

// Processing scheme: None = 5 user slots.
// FKv2 = calibrated PCHIP curve selected from the P35 background.
// FKv3 = recovered ML5 host flow: fitted arcsinh,wp=1 and paired log10gain.
// Element values keep their historical numbering: None=0, FKv2=2, FKv3=3.
// FKv1 (=1) was removed; instances saved with it fail the scheme check in
// ExecuteOn until the user re-selects a scheme. Never renumber to fill the
// gap - a stored value 2 must keep meaning FKv2.
class FKNormSchemeParameter : public pcl::MetaEnumeration
{
public:
   FKNormSchemeParameter( pcl::MetaProcess* P ) : pcl::MetaEnumeration( P )
   {
      TheFKNormSchemeParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "normScheme";
   }
   pcl::size_type NumberOfElements() const override
   {
      return 3;
   }
   pcl::IsoString ElementId( pcl::size_type idx ) const override
   {
      return (idx == 0) ? "None" : (idx == 1) ? "FKv2" : "FKv3";
   }
   int ElementValue( pcl::size_type idx ) const override
   {
      return (idx == 0) ? 0 : (idx == 1) ? 2 : 3;
   }
   pcl::size_type DefaultValueIndex() const override
   {
      return 2;
   }
};

class FKReflectPadParameter : public pcl::MetaBoolean
{
public:
   FKReflectPadParameter( pcl::MetaProcess* P ) : pcl::MetaBoolean( P )
   {
      TheFKReflectPadParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "reflectPad";
   }
};

class FKCorrectOnlyParameter : public pcl::MetaBoolean
{
public:
   FKCorrectOnlyParameter( pcl::MetaProcess* P ) : pcl::MetaBoolean( P )
   {
      TheFKCorrectOnlyParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "correctOnly";
   }
};

// Mode flag. Not a parameters-tensor column.
class FKLunarPlanetaryParameter : public pcl::MetaBoolean
{
public:
   FKLunarPlanetaryParameter( pcl::MetaProcess* P ) : pcl::MetaBoolean( P )
   {
      TheFKLunarPlanetaryParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "lunarPlanetary";
   }
};

class FKDeviceParameter : public pcl::MetaEnumeration
{
public:
   FKDeviceParameter( pcl::MetaProcess* P ) : pcl::MetaEnumeration( P )
   {
      TheFKDeviceParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "device";
   }
   pcl::size_type NumberOfElements() const override
   {
      return 3;
   }
   pcl::IsoString ElementId( pcl::size_type idx ) const override
   {
      switch ( idx )
      {
      case 0:  return "Auto";
      case 1:  return "GPU";
      default: return "CPU";
      }
   }
   int ElementValue( pcl::size_type idx ) const override
   {
      return int( idx );
   }
};

class FKVerboseParameter : public pcl::MetaBoolean
{
public:
   FKVerboseParameter( pcl::MetaProcess* P ) : pcl::MetaBoolean( P )
   {
      TheFKVerboseParameter = this;
   }
   pcl::IsoString Id() const override
   {
      return "verbose";
   }
};


// ---------------------------------------------------------------------------
// Process metaobject
// ---------------------------------------------------------------------------

class FKAstroInferenceProcess : public pcl::MetaProcess
{
public:

   FKAstroInferenceProcess();

   pcl::IsoString Id() const override;
   pcl::IsoString Aliases() const override;
   pcl::IsoString Categories() const override;
   pcl::IsoString IconImageSVG() const override;

   // View context only: never executable globally (no target image).
   bool CanProcessGlobal() const override
   {
      return false;
   }

   // Custom GUI so the process opens directly when activated.
   pcl::ProcessInterface* DefaultInterface() const override
   {
      return TheFKAstroInterface;
   }

   pcl::ProcessImplementation* Create() const override;
   pcl::ProcessImplementation* Clone( const pcl::ProcessImplementation& ) const override;
};

// ---------------------------------------------------------------------------
// Process instance
// ---------------------------------------------------------------------------

class FKAstroInferenceInstance : public pcl::ProcessImplementation
{
public:

   FKAstroInferenceInstance( const FKAstroInferenceProcess* );

   void Assign( const pcl::ProcessImplementation& ) override;
   bool Validate( pcl::String& info ) override;
   bool CanExecuteOn( const pcl::View& view, pcl::String& whyNot ) const override;
   bool ExecuteOn( pcl::View& view ) override;

   void* LockParameter( const pcl::MetaParameter* p, pcl::size_type tableRow ) override;
   bool AllocateParameter( pcl::size_type sizeOrLength, const pcl::MetaParameter* p, pcl::size_type tableRow ) override;
   pcl::size_type ParameterLength( const pcl::MetaParameter* p, pcl::size_type tableRow ) const override;

   // Process parameters
   pcl::String p_modelPath;
   double      p_sharpenStars      = 0.5;   // [0, 0.7]
   double      p_adjustStarHalos   = 0.0;   // [-0.5, 0.5]
   double      p_sharpenNonstellar = 0.5;   // [0, 1]
   bool        p_autoPSF      = true;
   double      p_psfDiameter  = 0.0;         // [0, 8], used when !p_autoPSF
   double      p_tileOverlap  = 0.2;         // fraction of tile size, [0, 0.5]
   bool        p_correctOnly  = true;
   // FKv3 mode: controls normalization and effective parameter overrides.
   bool        p_lunarPlanetary = false;
   pcl::pcl_enum p_normScheme = 3;    // 0=None 2=FKv2 3=FKv3 (1 = removed FKv1)
   bool        p_reflectPad   = true;
   pcl::pcl_enum p_device       = 0;  // 0=Auto 1=GPU 2=CPU
   bool        p_verbose      = true;
};
