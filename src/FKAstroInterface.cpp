/*
 * fk-astro — FKAstroInference process interface implementation
 */

#include "FKAstroInterface.h"
#include "FKAstroProcess.h"
#include <pcl/FileDialog.h>

FKAstroInterface* TheFKAstroInterface = nullptr;

// The combo box holds 3 items (None/FKv2/FKv3) while the stored scheme keeps
// its historical values (None=0, FKv2=2, FKv3=3). A stored FKv1 (1) displays
// as FKv3 here but still fails the scheme check on execution until the user
// re-selects.
static int SchemeComboIndex( int value )
{
   return (value == 0) ? 0 : (value == 2) ? 1 : 2;
}
static int SchemeComboValue( int index )
{
   return (index == 0) ? 0 : (index == 1) ? 2 : 3;
}

// ---------------------------------------------------------------------------
// Deferred GUI initialization
// ---------------------------------------------------------------------------

bool FKAstroInterface::Launch( const pcl::MetaProcess&,
                               const pcl::ProcessImplementation* instance,
                               bool& dynamic, unsigned& )
{
   dynamic = false;

   if ( GUI == nullptr )
   {
      GUI = new GUIData;

      // ---------------------------------------------------------------
      // Model
      // ---------------------------------------------------------------
      GUI->ModelBar.SetTitle( "Model" );
      GUI->ModelBar.OnToggleSection(
         (pcl::SectionBar::section_event_handler)&FKAstroInterface::e_Section, *this );
      GUI->ModelBar.SetSection( GUI->ModelSection );

      GUI->ModelPathLabel.SetText( "Model path:" );
      GUI->ModelPathEdit.SetToolTip(
         "<p>Path to your CoreML model (.mlpackage / .mlmodel). It must "
         "expose a fixed-shape 3-D float32 multi-array input and output; "
         "an additional parameters tensor is used automatically when "
         "present.</p>" );
      GUI->ModelPathEdit.OnEditCompleted(
         (pcl::Edit::edit_event_handler)&FKAstroInterface::e_EditCompleted, *this );

      GUI->BrowseButton.SetText( " ..." );
      GUI->BrowseButton.SetToolTip( "<p>Browse for a CoreML model file.</p>" );
      GUI->BrowseButton.OnClick(
         (pcl::Button::click_event_handler)&FKAstroInterface::e_BrowseClicked, *this );

      GUI->PathSizer.SetSpacing( 4 );
      GUI->PathSizer.Add( GUI->ModelPathLabel );
      GUI->PathSizer.Add( GUI->ModelPathEdit, 100 );
      GUI->PathSizer.Add( GUI->BrowseButton );





      GUI->ModelSizer.SetMargin( 4 );
      GUI->ModelSizer.SetSpacing( 4 );
      GUI->ModelSizer.Add( GUI->PathSizer );
      GUI->ModelSection.SetSizer( GUI->ModelSizer );

      // ---------------------------------------------------------------
      // Stellar Adjustments
      // ---------------------------------------------------------------
      GUI->StellarBar.SetTitle( "Stellar Adjustments" );
      GUI->StellarBar.OnToggleSection(
         (pcl::SectionBar::section_event_handler)&FKAstroInterface::e_Section, *this );
      GUI->StellarBar.SetSection( GUI->StellarSection );

      GUI->SharpenStars.label.SetText( "Sharpen stars:" );
      GUI->SharpenStars.SetRange( 0.0, 0.7 );
      GUI->SharpenStars.SetPrecision( 2 );
      GUI->SharpenStars.SetValue( m_sharpenStars );
      GUI->SharpenStars.SetToolTip(
         "<p>Star sharpening strength. Range 0.0 – 0.7, default 0.5.</p>" );
      GUI->SharpenStars.OnValueUpdated(
         (pcl::NumericEdit::value_event_handler)&FKAstroInterface::e_SharpenStars, *this );

      GUI->AdjustStarHalos.label.SetText( "Adjust star halos:" );
      GUI->AdjustStarHalos.SetRange( -0.5, 0.5 );
      GUI->AdjustStarHalos.SetPrecision( 2 );
      GUI->AdjustStarHalos.SetValue( m_adjustStarHalos );
      GUI->AdjustStarHalos.SetToolTip(
         "<p>Star halo correction. Range -0.5 – +0.5, default 0 "
         "(negative suppresses halos, positive brightens them).</p>" );
      GUI->AdjustStarHalos.OnValueUpdated(
         (pcl::NumericEdit::value_event_handler)&FKAstroInterface::e_AdjustStarHalos, *this );

      GUI->StellarSizer.SetMargin( 4 );
      GUI->StellarSizer.SetSpacing( 4 );
      GUI->StellarSizer.Add( GUI->SharpenStars );
      GUI->StellarSizer.Add( GUI->AdjustStarHalos );
      GUI->StellarSection.SetSizer( GUI->StellarSizer );

      // ---------------------------------------------------------------
      // Nonstellar Adjustments
      // ---------------------------------------------------------------
      GUI->NonstellarBar.SetTitle( "Nonstellar Adjustments" );
      GUI->NonstellarBar.OnToggleSection(
         (pcl::SectionBar::section_event_handler)&FKAstroInterface::e_Section, *this );
      GUI->NonstellarBar.SetSection( GUI->NonstellarSection );

      GUI->SharpenNonstellar.label.SetText( "Sharpen nonstellar:" );
      GUI->SharpenNonstellar.SetRange( 0.0, 1.0 );
      GUI->SharpenNonstellar.SetPrecision( 2 );
      GUI->SharpenNonstellar.SetValue( m_sharpenNonstellar );
      GUI->SharpenNonstellar.SetToolTip(
         "<p>Detail sharpening strength for nonstellar structures "
         "(galaxies, nebulae). Range 0.0 – 1.0, default 0.5.</p>" );
      GUI->SharpenNonstellar.OnValueUpdated(
         (pcl::NumericEdit::value_event_handler)&FKAstroInterface::e_SharpenNonstellar, *this );

      GUI->AutoPSFCheck.SetText( "Automatic PSF" );
      GUI->AutoPSFCheck.SetChecked( m_autoPSF );
      GUI->AutoPSFCheck.SetToolTip(
         "<p>When enabled, the model estimates the PSF (point spread "
         "function) automatically; the manual diameter control is "
         "disabled. Uncheck to specify a fixed PSF diameter.</p>" );
      GUI->AutoPSFCheck.OnCheck(
         (pcl::Button::check_event_handler)&FKAstroInterface::e_AutoPSF, *this );

      GUI->PSFDiameter.label.SetText( "PSF diameter (px):" );
      GUI->PSFDiameter.SetRange( 0.0, 8.0 );
      GUI->PSFDiameter.SetPrecision( 2 );
      GUI->PSFDiameter.SetValue( m_psfDiameter );
      GUI->PSFDiameter.Enable( !m_autoPSF );
      GUI->PSFDiameter.SetToolTip(
         "<p>Fixed PSF diameter in pixels for nonstellar structures. "
         "Only used when Automatic PSF is disabled. Range 0.0 – 8.0, "
         "default 0.0.</p>" );
      GUI->PSFDiameter.OnValueUpdated(
         (pcl::NumericEdit::value_event_handler)&FKAstroInterface::e_PSFDiameter, *this );

      GUI->NonstellarSizer.SetMargin( 4 );
      GUI->NonstellarSizer.SetSpacing( 4 );
      GUI->NonstellarSizer.Add( GUI->SharpenNonstellar );
      GUI->NonstellarSizer.Add( GUI->AutoPSFCheck );
      GUI->NonstellarSizer.Add( GUI->PSFDiameter );
      GUI->NonstellarSection.SetSizer( GUI->NonstellarSizer );

      // ---------------------------------------------------------------
      // Options
      // ---------------------------------------------------------------
      GUI->OptionsBar.SetTitle( "Options" );
      GUI->OptionsBar.OnToggleSection(
         (pcl::SectionBar::section_event_handler)&FKAstroInterface::e_Section, *this );
      GUI->OptionsBar.SetSection( GUI->OptionsSection );

      GUI->CorrectOnlyCheck.SetText( "Correct only" );
      GUI->CorrectOnlyCheck.SetChecked( m_correctOnly );
      GUI->CorrectOnlyCheck.SetToolTip(
         "<p>When enabled, the model is asked to only correct the PSF "
         "(no additional sharpening); when disabled, the sharpening "
         "controls apply.</p>" );
      GUI->CorrectOnlyCheck.OnCheck(
         (pcl::Button::check_event_handler)&FKAstroInterface::e_CorrectOnly, *this );

      GUI->LunarPlanetaryCheck.SetText( "Lunar/Planetary" );
      GUI->LunarPlanetaryCheck.SetChecked( m_lunarPlanetary );
      GUI->LunarPlanetaryCheck.SetToolTip(
         "<p>With FKv3, use the lunar/planetary normalization and a manual "
         "PSF diameter. Stellar adjustments and Automatic PSF are disabled. "
         "Correct Only overrides this mode.</p>" );
      GUI->LunarPlanetaryCheck.OnCheck(
         (pcl::Button::check_event_handler)&FKAstroInterface::e_LunarPlanetary, *this );

      GUI->NormLabel.SetText( "Processing scheme:" );
      GUI->NormCombo.AddItem( "None" );
      GUI->NormCombo.AddItem( "FKv2" );
      GUI->NormCombo.AddItem( "FKv3" );
      GUI->NormCombo.SetCurrentItem( SchemeComboIndex( m_normScheme ) );
      GUI->NormCombo.SetToolTip(
         "<p><b>None</b> — pixels are passed to the model unchanged and the "
         "parameters vector carries the five user slots (sharpen stars, "
         "star halos, PSF diameter, automatic PSF, sharpen nonstellar).</p>"
         "<p><b>FKv2</b> (legacy calibration) — the background statistic bg "
         "is measured once per image (P35 quantile) and the forward/inverse "
         "transforms use calibrated monotone curves captured at runtime from "
         "the ML5 model inputs (FKCurves.h): the curve whose bg is closest "
         "in log space is selected and evaluated with monotone PCHIP "
         "interpolation, with the inverse saturating at the curve endpoints. "
         "Out-of-range window pixels are zero-filled.</p>"
         "<p><b>FKv3</b> (default, recovered ML5 host flow) — fit an arcsinh "
         "transform to the whole "
         "image with white point 1. The sixth parameter uses the gain of "
         "this same transform. Tiles use zero padding and are blended before "
         "the inverse transform. Requires 1024x1024x3 input, 512x512x3 "
         "output and six parameters.</p>" );
      GUI->NormCombo.OnItemSelected(
         (pcl::ComboBox::item_event_handler)&FKAstroInterface::e_NormScheme, *this );

      GUI->NormSizer.SetSpacing( 4 );
      GUI->NormSizer.Add( GUI->NormLabel );
      GUI->NormSizer.Add( GUI->NormCombo );

      GUI->OptionsSizer.SetMargin( 4 );
      GUI->OptionsSizer.SetSpacing( 4 );
      GUI->OptionsSizer.Add( GUI->CorrectOnlyCheck );
      GUI->OptionsSizer.Add( GUI->LunarPlanetaryCheck );
      GUI->OptionsSizer.Add( GUI->NormSizer );
      GUI->OptionsSection.SetSizer( GUI->OptionsSizer );

      // ---------------------------------------------------------------
      // Additional Options
      // ---------------------------------------------------------------
      GUI->AdditionalBar.SetTitle( "Additional Options" );
      GUI->AdditionalBar.OnToggleSection(
         (pcl::SectionBar::section_event_handler)&FKAstroInterface::e_Section, *this );
      GUI->AdditionalBar.SetSection( GUI->AdditionalSection );

      GUI->TileOverlap.label.SetText( "Tile overlap:" );
      GUI->TileOverlap.SetRange( 0.0, 0.5 );
      GUI->TileOverlap.SetPrecision( 2 );
      GUI->TileOverlap.SetValue( m_tileOverlap );
      GUI->TileOverlap.SetToolTip(
         "<p>Overlap fraction, range 0.0 – 0.5, default 0.2. FKv3 uses "
         "this value for both the tile count and cosine transition width.</p>" );
      GUI->TileOverlap.OnValueUpdated(
         (pcl::NumericEdit::value_event_handler)&FKAstroInterface::e_Overlap, *this );

      GUI->ReflectPadCheck.SetText( "Reflect border padding" );
      GUI->ReflectPadCheck.SetChecked( m_reflectPad );
      GUI->ReflectPadCheck.SetToolTip(
         "<p>Mirror the image at borders when a tile window extends past "
         "the edge. Unchecked: replicate edge pixels. FKv3 always uses "
         "zero padding.</p>" );
      GUI->ReflectPadCheck.OnCheck(
         (pcl::Button::check_event_handler)&FKAstroInterface::e_ReflectPad, *this );

      GUI->DeviceLabel.SetText( "Compute device:" );
      GUI->DeviceCombo.AddItem( "Automatic (GPU)" );
      GUI->DeviceCombo.AddItem( "GPU" );
      GUI->DeviceCombo.AddItem( "CPU" );
      GUI->DeviceCombo.SetCurrentItem( m_device );
      GUI->DeviceCombo.OnItemSelected(
         (pcl::ComboBox::item_event_handler)&FKAstroInterface::e_Device, *this );

      GUI->DeviceSizer.SetSpacing( 4 );
      GUI->DeviceSizer.Add( GUI->DeviceLabel );
      GUI->DeviceSizer.Add( GUI->DeviceCombo );

      GUI->VerboseCheck.SetText( "Verbose console output" );
      GUI->VerboseCheck.SetChecked( m_verbose );
      GUI->VerboseCheck.OnCheck(
         (pcl::Button::check_event_handler)&FKAstroInterface::e_Verbose, *this );

      GUI->AdditionalSizer.SetMargin( 4 );
      GUI->AdditionalSizer.SetSpacing( 4 );
      GUI->AdditionalSizer.Add( GUI->TileOverlap );
      GUI->AdditionalSizer.Add( GUI->ReflectPadCheck );
      GUI->AdditionalSizer.Add( GUI->DeviceSizer );
      GUI->AdditionalSizer.Add( GUI->VerboseCheck );
      GUI->AdditionalSection.SetSizer( GUI->AdditionalSizer );

      // ---------------------------------------------------------------
      // Root layout
      // ---------------------------------------------------------------
      GUI->VSizer.SetMargin( 4 );
      GUI->VSizer.SetSpacing( 4 );
      GUI->VSizer.Add( GUI->ModelBar );
      GUI->VSizer.Add( GUI->ModelSection );
      GUI->VSizer.Add( GUI->StellarBar );
      GUI->VSizer.Add( GUI->StellarSection );
      GUI->VSizer.Add( GUI->NonstellarBar );
      GUI->VSizer.Add( GUI->NonstellarSection );
      GUI->VSizer.Add( GUI->OptionsBar );
      GUI->VSizer.Add( GUI->OptionsSection );
      GUI->VSizer.Add( GUI->AdditionalBar );
      GUI->VSizer.Add( GUI->AdditionalSection );

      SetSizer( GUI->VSizer );
      AdjustToContents();
      UpdateEnabledStates();
   }

   if ( instance != nullptr )
      ImportProcess( *instance );

   return true;
}

// ---------------------------------------------------------------------------
// ProcessInterface
// ---------------------------------------------------------------------------

pcl::IsoString FKAstroInterface::Id() const
{
   return "FKAstroInterface";
}

pcl::MetaProcess* FKAstroInterface::Process() const
{
   return TheFKAstroInferenceProcess;
}

pcl::ProcessImplementation* FKAstroInterface::NewProcess() const
{
   FKAstroInferenceInstance* instance =
      new FKAstroInferenceInstance( TheFKAstroInferenceProcess );
   instance->p_modelPath         = m_modelPath;
   instance->p_sharpenStars      = m_sharpenStars;
   instance->p_adjustStarHalos   = m_adjustStarHalos;
   instance->p_sharpenNonstellar = m_sharpenNonstellar;
   instance->p_autoPSF           = m_autoPSF;
   instance->p_psfDiameter       = m_psfDiameter;
   instance->p_correctOnly       = m_correctOnly;
   instance->p_lunarPlanetary    = m_lunarPlanetary;
   instance->p_normScheme        = m_normScheme;
   instance->p_reflectPad        = m_reflectPad;
   instance->p_tileOverlap       = m_tileOverlap;
   instance->p_device            = m_device;
   instance->p_verbose           = m_verbose;
   return instance;
}

bool FKAstroInterface::ImportProcess( const pcl::ProcessImplementation& P )
{
   const FKAstroInferenceInstance* instance =
      dynamic_cast<const FKAstroInferenceInstance*>( &P );
   if ( instance == nullptr )
      return false;

   m_modelPath         = instance->p_modelPath;
   m_sharpenStars      = instance->p_sharpenStars;
   m_adjustStarHalos   = instance->p_adjustStarHalos;
   m_sharpenNonstellar = instance->p_sharpenNonstellar;
   m_autoPSF           = instance->p_autoPSF;
   m_psfDiameter       = instance->p_psfDiameter;
   m_correctOnly       = instance->p_correctOnly;
   m_lunarPlanetary    = instance->p_lunarPlanetary;
   m_normScheme        = int( instance->p_normScheme );
   m_reflectPad        = instance->p_reflectPad;
   m_tileOverlap       = instance->p_tileOverlap;
   m_device            = int( instance->p_device );
   m_verbose           = instance->p_verbose;

   if ( GUI != nullptr )
   {
      GUI->ModelPathEdit.SetText( m_modelPath );
      GUI->SharpenStars.SetValue( m_sharpenStars );
      GUI->AdjustStarHalos.SetValue( m_adjustStarHalos );
      GUI->SharpenNonstellar.SetValue( m_sharpenNonstellar );
      GUI->AutoPSFCheck.SetChecked( m_autoPSF );
      GUI->PSFDiameter.SetValue( m_psfDiameter );
      GUI->PSFDiameter.Enable( !m_autoPSF );
      GUI->CorrectOnlyCheck.SetChecked( m_correctOnly );
      GUI->LunarPlanetaryCheck.SetChecked( m_lunarPlanetary );
      GUI->NormCombo.SetCurrentItem( SchemeComboIndex( m_normScheme ) );
      GUI->ReflectPadCheck.SetChecked( m_reflectPad );
      GUI->TileOverlap.SetValue( m_tileOverlap );
      GUI->DeviceCombo.SetCurrentItem( m_device );
      GUI->VerboseCheck.SetChecked( m_verbose );
      UpdateEnabledStates();
   }
   return true;
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

void FKAstroInterface::e_EditCompleted( pcl::Edit& sender )
{
   m_modelPath = sender.Text().Trimmed();
}

void FKAstroInterface::e_BrowseClicked( pcl::Button&, bool )
{
   pcl::OpenFileDialog d;
   d.SetCaption( "Select CoreML Model" );
   pcl::FileFilter coreMLFilter( "CoreML models", pcl::StringList()
      << ".mlpackage" << ".mlmodel" << ".mlpackage/" << ".mlmodel/" );
   pcl::FileFilter allFilter( "All files", "*" );
   d.SetFilters( pcl::FileDialog::filter_list() << coreMLFilter << allFilter );
   if ( !m_modelPath.IsEmpty() )
      d.SetInitialPath( m_modelPath );
   if ( d.Execute() )
   {
      m_modelPath = d.FileName();
      if ( GUI != nullptr )
         GUI->ModelPathEdit.SetText( m_modelPath );
   }
}

void FKAstroInterface::e_SharpenStars( pcl::NumericEdit&, double value )
{
   m_sharpenStars = value;
}

void FKAstroInterface::e_AdjustStarHalos( pcl::NumericEdit&, double value )
{
   m_adjustStarHalos = value;
}

void FKAstroInterface::e_SharpenNonstellar( pcl::NumericEdit&, double value )
{
   m_sharpenNonstellar = value;
}

void FKAstroInterface::e_AutoPSF( pcl::Button&, pcl::Button::check_state state )
{
   m_autoPSF = (state == pcl::CheckState::Checked);
   UpdateEnabledStates();
}

void FKAstroInterface::e_PSFDiameter( pcl::NumericEdit&, double value )
{
   m_psfDiameter = value;
}

void FKAstroInterface::e_CorrectOnly( pcl::Button&, pcl::Button::check_state state )
{
   m_correctOnly = (state == pcl::CheckState::Checked);
   UpdateEnabledStates();
}

void FKAstroInterface::e_LunarPlanetary( pcl::Button&, pcl::Button::check_state state )
{
   m_lunarPlanetary = (state == pcl::CheckState::Checked);
   UpdateEnabledStates();
}





void FKAstroInterface::e_Overlap( pcl::NumericEdit&, double value )
{
   m_tileOverlap = value;
}

void FKAstroInterface::e_Device( pcl::ComboBox&, int itemIndex )
{
   m_device = itemIndex;
}

void FKAstroInterface::e_Verbose( pcl::Button&, pcl::Button::check_state state )
{
   m_verbose = (state == pcl::CheckState::Checked);
}

void FKAstroInterface::e_Section( pcl::SectionBar&, pcl::Control&, bool start )
{
   if ( !start )
   {
      AdjustToContents();
      Update();
   }
}

void FKAstroInterface::UpdateEnabledStates()
{
   if ( GUI == nullptr )
      return;
   const bool adjustmentsEnabled = !m_correctOnly;
   const bool fkv3 = m_normScheme == 3;
   const bool lunar = fkv3 && m_lunarPlanetary && adjustmentsEnabled;
   GUI->SharpenStars.Enable( adjustmentsEnabled && !lunar );
   GUI->AdjustStarHalos.Enable( adjustmentsEnabled && !lunar );
   GUI->SharpenNonstellar.Enable( adjustmentsEnabled );
   GUI->AutoPSFCheck.Enable( adjustmentsEnabled && !lunar );
   GUI->AutoPSFCheck.SetChecked( !adjustmentsEnabled || ( !lunar && m_autoPSF ) );
   GUI->PSFDiameter.Enable( adjustmentsEnabled && ( lunar || !m_autoPSF ) );
   GUI->LunarPlanetaryCheck.Enable( adjustmentsEnabled && fkv3 );
   GUI->LunarPlanetaryCheck.SetChecked( fkv3 && adjustmentsEnabled && m_lunarPlanetary );
   GUI->ReflectPadCheck.Enable( !fkv3 && m_normScheme != 2 );
   GUI->ReflectPadCheck.SetChecked( !fkv3 && m_normScheme != 2 && m_reflectPad );
}

// ---------------------------------------------------------------------------
// Normalization / padding event handlers
// ---------------------------------------------------------------------------

void FKAstroInterface::e_NormScheme( pcl::ComboBox&, int itemIndex )
{
   m_normScheme = SchemeComboValue( itemIndex );
   UpdateEnabledStates();
}

void FKAstroInterface::e_ReflectPad( pcl::Button&, pcl::Button::check_state state )
{
   m_reflectPad = (state == pcl::CheckState::Checked);
}
