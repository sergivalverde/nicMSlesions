#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#define DEBUG false  // set to true to see (print) how cost function evolves for registrations and bias field correction

#ifdef _WIN32
#define ATLASFILENAME ".\\ref_vols\\atlas.nii.gz"
#define MASKFILENAME ".\\ref_vols\\atlas_mask.nii.gz"
#define ERODEDMASKFILENAME ".\\ref_vols\\atlas_mask_eroded.nii.gz"
#define DILATEDMASKFILENAME ".\\ref_vols\\atlas_mask_dilated.nii.gz"
#define INITIALMASKFILENAME ".\\ref_vols\\Mini.nii.gz"
#define PATH_SEPARATOR "\\"
#define PATH_SEPARATOR_CHAR '\\'
#define DEL_CMD "del /Q "
#define MOVE_CMD "move "
#else
#define ATLASFILENAME "./ref_vols/atlas.nii.gz"
#define MASKFILENAME "./ref_vols/atlas_mask.nii.gz"
#define ERODEDMASKFILENAME "./ref_vols/atlas_mask_eroded.nii.gz"
#define DILATEDMASKFILENAME "./ref_vols/atlas_mask_dilated.nii.gz"
#define INITIALMASKFILENAME "./ref_vols/Mini.nii.gz"
#define PATH_SEPARATOR "/"
#define PATH_SEPARATOR_CHAR '/'
#define DEL_CMD "rm -f "
#define MOVE_CMD "mv "
#endif



// DO NOT TOUCH THESE DEFINE:S
#define MINI_PIXS 624752
#define PIXSIZE 1.5						// pixel size of the atlas
#define NL 3237							// number of landmarks
#define N_MODES 19						// modes of shape model
#define NB2_MAX 36.1909 				// 99% pctile chi-square
#define N_FACES 6470					// number of faces in the model
#define SEARCH_DIST (floor(20/PIXSIZE))
#define INC 0.5							// make sure SEARCH_DIST/INC is integer!
#define OUT_CONSTANT -1000				// out of buffer constant, negative and large 
#define NODE_TERMINAL -1
#define MAX_R 30000



// Parameters you can touch (although I wouldn't recommended it...)
#define SHRINK_FACTOR_BFC 3				// must be integer
#define SIGMA_R 0.75					// sigma to blur likelihood map
#define EPS 0.001						// background for cost image
#define PREC_B (20.0/PIXSIZE) 			// for exhaustive search 20/pixdimAtlas(1);
#define PREC_B_FINE 1					// for gradient descent
#define MAX_IT_ALL 10					// iterations with exhaustive seach
#define STEP_INI 1						// for optimization of shape model
#define DELTA_B 1.0						// for optimization of shape model
#define MAX_IT_IN 10					// for Newton's method
#define MAX_IT 50						// for Newton's method
#define DELTA  0.1						// tolerance




// system includes
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <list>
#include <string>
#include <cctype>
#include <vector>
using namespace std;


// ITK includes
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include "itkArray.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include "itkCommand.h"
#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkTranslationTransform.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkMultiResolutionPyramidImageFilter.h"
#include "itkCenteredTransformInitializer.h"
#include "itkCenteredAffineTransform.h"
#include "itkSimilarity3DTransform.h"
#include "itkAffineTransform.h"
#include "itkImageMaskSpatialObject.h"
#include "itkResampleImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkDerivativeImageFilter.h"
#include "itkSquareImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkSqrtImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkBSplineControlPointImageFilter.h"
#include "itkExpImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkShiftScaleImageFilter.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkConnectedThresholdImageFilter.h"
#include "itkOrImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkConstantPadImageFilter.h"
#include "itkCropImageFilter.h"
#include "itkN4BiasFieldCorrectionImageFilter.h"

// own includes
#include "aux_methods.h"
#include "loadModel.h"
#include "RobustStrip.h"




// Observer for the registration

template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command
{
public:
  typedef  RegistrationInterfaceCommand   Self;
  typedef  itk::Command                   Superclass;
  typedef  itk::SmartPointer<Self>        Pointer;
  itkNewMacro( Self );

protected:
  RegistrationInterfaceCommand() {};

public:
  typedef   TRegistration                              RegistrationType;
  typedef   RegistrationType *                         RegistrationPointer;
  typedef   itk::RegularStepGradientDescentOptimizer   OptimizerType;
  typedef   OptimizerType *                            OptimizerPointer;

  void Execute(itk::Object * object, const itk::EventObject & event)
    {
    if( !(itk::IterationEvent().CheckEvent( &event )) )
      {
      return;
      }
    RegistrationPointer registration =
      dynamic_cast<RegistrationPointer>( object );

    OptimizerPointer optimizer = dynamic_cast< OptimizerPointer >(
      registration->GetModifiableOptimizer() );

    if (DEBUG){
      std::cout << "-------------------------------------" << std::endl;
      std::cout << "MultiResolution Level : "
                << registration->GetCurrentLevel()  << std::endl;
      std::cout << std::endl << std::flush;
    }
 
    if ( registration->GetCurrentLevel() == 0 )
      {

      if(typeid( *(registration->GetTransform()) ) == typeid( itk::Similarity3DTransform<double> ))  // if similarity transform
        {
        optimizer->SetMaximumStepLength( 0.5 );
        }
      else
        {
        optimizer->SetMaximumStepLength( 0.25 );
        }
      optimizer->SetMinimumStepLength( 0.01 );
      }
    else
      {
      optimizer->SetMaximumStepLength( optimizer->GetMaximumStepLength() / 4.0 );
      optimizer->SetMinimumStepLength( optimizer->GetMinimumStepLength() / 10.0 );
      }
    }

  void Execute(const itk::Object * , const itk::EventObject & )
    { return; }
};





template<class TFilter>
class N4CommandIterationUpdate : public itk::Command
{
public:
  typedef N4CommandIterationUpdate   Self;
  typedef itk::Command             Superclass;
  typedef itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );
protected:
  N4CommandIterationUpdate() {};
public:

  void Execute(itk::Object *caller, const itk::EventObject & event)
    {
    Execute( (const itk::Object *) caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
    const TFilter * filter =
      dynamic_cast< const TFilter * >( object );
    if( typeid( event ) != typeid( itk::IterationEvent ) )
      { return; }

    if (DEBUG){
      std::cout << "Iteration " << filter->GetElapsedIterations()
        << " (of " << filter->GetMaximumNumberOfIterations() << ").  ";
      std::cout << " Current convergence value = "
        << filter->GetCurrentConvergenceMeasurement()
        << " (threshold = " << filter->GetConvergenceThreshold()
        << ")" << std::endl << std::flush;
    }
    }

};




//  The following section of code implements an observer
//  that will monitor the evolution of the registration process.
//
class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef  itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );

protected:
  CommandIterationUpdate() {};

public:
  typedef   itk::RegularStepGradientDescentOptimizer OptimizerType;
  typedef   const OptimizerType *                    OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
    {
      Execute( (const itk::Object *)caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
      OptimizerPointer optimizer =
        dynamic_cast< OptimizerPointer >( object );
      if( !(itk::IterationEvent().CheckEvent( &event )) )
        {
        return;
        }
    if (DEBUG){
        std::cout << optimizer->GetCurrentIteration() << "   ";
        std::cout << optimizer->GetValue() << "   ";
        std::cout << optimizer->GetCurrentPosition() << std::endl << std::flush;
      }
    }
};



////////
// Main function
///////


int main( int argc, char ** argv )
{

	try{

		if( argc < 3 )
		{
			cerr << "Usage: " << endl;
			cerr << argv[0] << " inputImageFile  strippedImageFile [maskImageFile] [seed]" << endl;
			return EXIT_FAILURE;
		}


		char * inputFilename  = argv[1];
		char * outputFilename = argv[2];

		typedef  float      PixelType;
		typedef  unsigned char      MaskPixelType;
		const   unsigned int   Dimension = 3;

		typedef itk::Image< MaskPixelType, Dimension >    MaskImageType;
		typedef itk::Image< PixelType, Dimension >    ImageType;
		typedef itk::Image< PixelType, 2 >    Image2DType;
		typedef itk::ImageFileReader< MaskImageType >  MaskReaderType;
		typedef itk::ImageFileReader< ImageType >  ReaderType;
		typedef itk::ImageFileReader< Image2DType >  Reader2DType;
		typedef itk::ImageFileWriter< ImageType >  WriterType;
		typedef itk::ImageFileWriter< Image2DType >  Writer2DType;
		typedef itk::ImageFileWriter< MaskImageType >  MaskWriterType;

		typedef itk::TranslationTransform< double, Dimension > TranslationTransformType;
		typedef itk::Similarity3DTransform< double > SimilarityTransformType;
		typedef itk::AffineTransform< double, Dimension > AffineTransformType;
		typedef itk::CenteredAffineTransform< double, Dimension > CenteredAffineTransformType;
		typedef itk::RegularStepGradientDescentOptimizer       OptimizerType;
		typedef itk::LinearInterpolateImageFunction<ImageType, double> LinearInterpolatorType;
		typedef itk::NearestNeighborInterpolateImageFunction<ImageType, double> NNInterpolatorType;
		typedef itk::NearestNeighborInterpolateImageFunction<MaskImageType, double> MaskNNInterpolatorType;
		typedef itk::BSplineInterpolateImageFunction<ImageType, double,double> BSplineInterpolatorType;
		typedef itk::MattesMutualInformationImageToImageMetric<ImageType,ImageType >   MetricType;
		typedef itk::MultiResolutionImageRegistrationMethod< ImageType, ImageType >   RegistrationType;
		typedef RegistrationType::ParametersType ParametersType;
		typedef itk::MultiResolutionPyramidImageFilter<ImageType, ImageType >   FixedImagePyramidType;
		typedef itk::MultiResolutionPyramidImageFilter<ImageType,ImageType >   MovingImagePyramidType;
		typedef itk::CenteredTransformInitializer<TranslationTransformType,ImageType,ImageType > TranslationTransformInitializerType;
		typedef itk::CenteredTransformInitializer<CenteredAffineTransformType,ImageType,ImageType > AffineTransformInitializerType;
		typedef itk::CenteredTransformInitializer<SimilarityTransformType,ImageType,ImageType > SimilarityTransformInitializerType;
		typedef CenteredAffineTransformType::ParametersType CATparametersType;
		typedef itk::ImageMaskSpatialObject< Dimension > MaskType;
		typedef itk::ResampleImageFilter< ImageType,ImageType >    ResampleFilterType;
		typedef itk::ResampleImageFilter< MaskImageType,MaskImageType >    MaskResampleFilterType;
		typedef itk::MaskImageFilter<ImageType, MaskImageType, ImageType> MaskImageFilterType;
		typedef itk::DivideImageFilter<ImageType,ImageType,ImageType> DivideImageFilterType;
		typedef itk::DivideImageFilter<ImageType,ImageType,ImageType> DivideImageFilterType;
		typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> GaussianFilterType;
		typedef itk::DerivativeImageFilter<ImageType,ImageType> DerivativeFilterType;
		typedef itk::SquareImageFilter<ImageType,ImageType> SquareFilterType;
		typedef itk::AddImageFilter<ImageType,ImageType,ImageType> AddFilterType;
		typedef itk::SqrtImageFilter<ImageType,ImageType> SqrtFilterType;
		typedef itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;
		typedef itk::ImageRegionIteratorWithIndex< MaskImageType > MaskIteratorType;
		typedef itk::ShiftScaleImageFilter<ImageType,ImageType> ShiftScaleFilterType;
		typedef itk::BinaryBallStructuringElement<MaskPixelType,Dimension> BallType;
		typedef itk::BinaryErodeImageFilter<MaskImageType,MaskImageType,BallType> ErodeFilterType;
		typedef itk::BinaryDilateImageFilter<MaskImageType,MaskImageType,BallType> DilateFilterType;
		typedef itk::ConnectedThresholdImageFilter<MaskImageType,MaskImageType> ConnectedFilterType;
		typedef itk::OrImageFilter<MaskImageType,MaskImageType,MaskImageType> OrFilterType;
		typedef itk::ShrinkImageFilter<ImageType, ImageType> ShrinkerType;
		typedef itk::ShrinkImageFilter<MaskImageType, MaskImageType> MaskShrinkerType;
		typedef itk::ConstantPadImageFilter<MaskImageType, MaskImageType> PadFilterType;
		typedef itk::CropImageFilter<MaskImageType, MaskImageType> CropFilterType;
		typedef itk::N4BiasFieldCorrectionImageFilter<ImageType, MaskImageType,ImageType> CorrecterType;
		typedef itk::BSplineControlPointImageFilter<CorrecterType::BiasFieldControlPointLatticeType,   CorrecterType::ScalarImageType> BSplinerType;
		typedef itk::ExpImageFilter<ImageType, ImageType> ExpFilterType;
		typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideImageFilterType;
		typedef RegistrationInterfaceCommand<RegistrationType> CommandType;

		WriterType::Pointer writer = WriterType::New();  // this a Swiss Army writer I use throughout the code; I'd rather declare it here...

		// seed random number generator, if necessary
		if(argc>4) srand(atoi(argv[4]));

		// temporary directories
		char temp_dir_base[5000]; 
		int posLastSlash=-1;
		for(int kkk=0; kkk<strlen(argv[2]); kkk++)
		{
			if(argv[2][kkk]==PATH_SEPARATOR_CHAR)	
				posLastSlash=kkk;
		}
		if(posLastSlash==-1)
			sprintf(temp_dir_base,".%stemp_%s",PATH_SEPARATOR,argv[2]);
		else{
			for(int kkk=0; kkk<posLastSlash; kkk++)
				temp_dir_base[kkk]=argv[2][kkk];
			temp_dir_base[posLastSlash]='\0';
			strcat(temp_dir_base,PATH_SEPARATOR);
			strcat(temp_dir_base,"temp_");
			strcat(temp_dir_base,argv[2]+posLastSlash+1);
  
		} 


		char command[5000];
		sprintf(command,"mkdir %s",temp_dir_base); 
		system(command);

		char *bgName=generateBogusFilename();
                char temp_dir[2000];
		sprintf(temp_dir,"%s%s%s",temp_dir_base,PATH_SEPARATOR,bgName);
		char trash_file[2000];
 		sprintf(trash_file,"%s%s%s",temp_dir,PATH_SEPARATOR,"trash.txt");
		free(bgName);


		sprintf(command,"mkdir %s",temp_dir);
		system(command);
		sprintf(command,"%s%s%s*.*",DEL_CMD,temp_dir,PATH_SEPARATOR);
		system(command);

		time_t t_ini_global=time(NULL);
		time_t t_ini_step=0;
		char filename[5000];



		/*******************************/
		/*     Step 1: Read images     */
		/*******************************/

		std::cout << "Step 1  of 9: reading in images..."  << std::flush;
		t_ini_step = time(NULL); 


		ReaderType::Pointer readerI = ReaderType::New();
		readerI->SetFileName( inputFilename  );
		readerI->Update(); 
		ReaderType::Pointer readerAtlas = ReaderType::New();
		readerAtlas->SetFileName( ATLASFILENAME );
		readerAtlas->Update();
		MaskReaderType::Pointer readerM = MaskReaderType::New();
		readerM->SetFileName( MASKFILENAME  );
		readerM->Update();
		MaskReaderType::Pointer readerMe = MaskReaderType::New();
		readerMe->SetFileName( ERODEDMASKFILENAME  );
		readerMe->Update();
		MaskReaderType::Pointer readerMd = MaskReaderType::New();
		readerMd->SetFileName( DILATEDMASKFILENAME  );
		readerMd->Update();
		MaskReaderType::Pointer readerMini = MaskReaderType::New();
		readerMini->SetFileName( INITIALMASKFILENAME  );
		readerMini->Update();

		// Make sure minimum is not negative...
		IteratorType It_1( readerI->GetOutput(), readerI->GetOutput()->GetRequestedRegion() );
		PixelType aux1, min;
		min=1e10;
		for ( It_1.GoToBegin(); !It_1.IsAtEnd(); ++It_1)
		{
			aux1=It_1.Get();
			if(aux1<min) min=aux1;
		}
		if(min<10){
			for ( It_1.GoToBegin(); !It_1.IsAtEnd(); ++It_1)
			{
				aux1=It_1.Get();
				It_1.Set(aux1-min+10);
			}

		}

		printf("Done! It took roughly %d seconds\n",(int)(time(NULL)-t_ini_step));
		std::cout << std::flush;




		/*******************************/
		/*     Step 2: Registration    */
		/*******************************/

		std::cout << "Step 2 of 9: registration..."  << std::endl << std::flush;
		t_ini_step = time(NULL);

		std::cout << "  2a) Similarity transform..."<< std::endl << std::flush;


		SimilarityTransformType::Pointer      similarityTransform     = SimilarityTransformType::New();
		OptimizerType::Pointer      optimizer     = OptimizerType::New();
		LinearInterpolatorType::Pointer   linearInterpolator  = LinearInterpolatorType::New();
		RegistrationType::Pointer   registration  = RegistrationType::New();
		MetricType::Pointer         metric        = MetricType::New();


		FixedImagePyramidType::Pointer fixedImagePyramid = FixedImagePyramidType::New();
		MovingImagePyramidType::Pointer movingImagePyramid =  MovingImagePyramidType::New();
		registration->SetOptimizer(     optimizer     );
  		registration->SetTransform(     similarityTransform     );
  		registration->SetInterpolator(  linearInterpolator  );
  		registration->SetMetric( metric  );
  		registration->SetFixedImagePyramid( fixedImagePyramid );
  		registration->SetMovingImagePyramid( movingImagePyramid );
 		registration->SetFixedImage(    readerAtlas->GetOutput()    );
  		registration->SetMovingImage(   readerI->GetOutput()   );
  		registration->SetFixedImageRegion( readerAtlas->GetOutput()->GetBufferedRegion() );

		SimilarityTransformInitializerType::Pointer initializer = SimilarityTransformInitializerType::New();
		initializer->SetTransform( similarityTransform );
		initializer->SetFixedImage( readerAtlas->GetOutput() );
		initializer->SetMovingImage( readerI->GetOutput()  );
		initializer->MomentsOn();
		initializer->InitializeTransform();
		registration->SetInitialTransformParameters( similarityTransform->GetParameters() );

		OptimizerType::ScalesType  optimizerScales( similarityTransform->GetNumberOfParameters() );
		const double translationScale = 1.0 / 10000.0;
  		optimizerScales[0] = 1.0;
  		optimizerScales[1] = 1.0;
  		optimizerScales[2] = 1.0;
  		optimizerScales[3] = translationScale;
  		optimizerScales[4] = translationScale;
  		optimizerScales[5] = translationScale;
  		optimizerScales[6] = 1.0;
  		optimizer->SetScales( optimizerScales );

		metric->SetNumberOfHistogramBins( 32 );
		metric->SetNumberOfSpatialSamples( 65000 );

		optimizer->SetNumberOfIterations( 66 );
		optimizer->SetRelaxationFactor( 0.9 );

  		CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  		optimizer->AddObserver( itk::IterationEvent(), observer );
  		CommandType::Pointer commander = CommandType::New();
  		registration->AddObserver( itk::IterationEvent(), commander );

  		registration->SetNumberOfLevels( 3 );
  		try
    		{
    			registration->Update();
			if (DEBUG){
	    			 std::cout << "Optimizer stop condition: "
	              			<< registration->GetOptimizer()->GetStopConditionDescription()
                			<< std::endl << std::flush;
			}
    		}
  		catch( itk::ExceptionObject & err )
    		{
    			std::cout << "ExceptionObject caught in registration !" << std::endl;
    			std::cout << err << std::endl << std::flush;
    			return EXIT_FAILURE;
    		}

  		ParametersType finalParameters = registration->GetLastTransformParameters();
  		SimilarityTransformType::Pointer finalTransform = SimilarityTransformType::New();
  		finalTransform->SetParameters( finalParameters );
  		finalTransform->SetFixedParameters( similarityTransform->GetFixedParameters() );



		std::cout << "  2b) Affine transform..."<< std::endl << std::flush;

		AffineTransformType::Pointer      affineTransform     = AffineTransformType::New();
		optimizer     = OptimizerType::New();
		linearInterpolator  = LinearInterpolatorType::New();
		registration  = RegistrationType::New();
		metric        = MetricType::New();
		fixedImagePyramid =   FixedImagePyramidType::New();
		 movingImagePyramid = MovingImagePyramidType::New();

		registration->SetOptimizer(     optimizer     );
  		registration->SetTransform(     affineTransform     );
  		registration->SetInterpolator(  linearInterpolator  );
  		registration->SetMetric( metric  );
  		registration->SetFixedImagePyramid( fixedImagePyramid );
  		registration->SetMovingImagePyramid( movingImagePyramid );
 		registration->SetFixedImage(    readerAtlas->GetOutput()    );
  		registration->SetMovingImage(   readerI->GetOutput()   );
  		registration->SetFixedImageRegion(
	       	readerAtlas->GetOutput()->GetBufferedRegion() );

		affineTransform->SetMatrix(finalTransform->GetMatrix());
		affineTransform->SetOffset(finalTransform->GetOffset());
		registration->SetInitialTransformParameters( affineTransform->GetParameters() );


  		OptimizerType::ScalesType  optimizerScalesAffine( affineTransform->GetNumberOfParameters() );
  		optimizerScalesAffine[0] = 1.0;
  		optimizerScalesAffine[1] = 1.0;
  		optimizerScalesAffine[2] = 1.0;
  		optimizerScalesAffine[3] = 1.0;
  		optimizerScalesAffine[4] = 1.0;
  		optimizerScalesAffine[5] = 1.0;
  		optimizerScalesAffine[6] = 1.0;
  		optimizerScalesAffine[7] = 1.0;
  		optimizerScalesAffine[8] = 1.0;
  		optimizerScalesAffine[9] = translationScale;
  		optimizerScalesAffine[10] = translationScale;
  		optimizerScalesAffine[11] = translationScale;
  		optimizer->SetScales( optimizerScalesAffine );

		metric->SetNumberOfHistogramBins( 32 );
  		metric->SetNumberOfSpatialSamples( 132000 );

		MaskType::Pointer spatialObjectMask = MaskType::New();
		spatialObjectMask->SetImage( readerMd->GetOutput() );
		metric->SetFixedImageMask(spatialObjectMask);

  		optimizer->SetNumberOfIterations( 66 );
  		optimizer->SetRelaxationFactor( 0.9 );

   		observer = CommandIterationUpdate::New();
  		optimizer->AddObserver( itk::IterationEvent(), observer );
  		commander = CommandType::New();
  		registration->AddObserver( itk::IterationEvent(), commander );

  		registration->SetNumberOfLevels( 3 );
		try
    		{
    			registration->Update();
			if (DEBUG){
	   			std::cout << "Optimizer stop condition: "
	              			<< registration->GetOptimizer()->GetStopConditionDescription()
              				<< std::endl << std::flush;
			}
		}
  		catch( itk::ExceptionObject & err )
    		{
    			std::cout << "ExceptionObject caught !" << std::endl;
    			std::cout << err << std::endl << std::flush;
    			return EXIT_FAILURE;
    		}

		finalParameters = registration->GetLastTransformParameters();
  		AffineTransformType::Pointer finalTransformAffine = AffineTransformType::New();
		finalTransformAffine->SetParameters( finalParameters );
		finalTransformAffine->SetFixedParameters( affineTransform->GetFixedParameters() );


		std::cout << "  2c) Resampling volume..."<< std::endl << std::flush;

  		ResampleFilterType::Pointer resample = ResampleFilterType::New();
		BSplineInterpolatorType::Pointer bsplineInterpolator = BSplineInterpolatorType::New();
		bsplineInterpolator->SetSplineOrder(3);

  		resample->SetTransform( finalTransformAffine );
  		resample->SetInput( readerI->GetOutput() );
		resample->SetSize(    readerAtlas->GetOutput()->GetLargestPossibleRegion().GetSize() );
  		resample->SetOutputOrigin(  readerAtlas->GetOutput()->GetOrigin() );
  		resample->SetOutputSpacing( readerAtlas->GetOutput()->GetSpacing() );
  		resample->SetOutputDirection( readerAtlas->GetOutput()->GetDirection() );
  		resample->SetDefaultPixelValue( -1000 );
		resample->SetInterpolator(bsplineInterpolator);
   		resample->Update();

		ImageType::Pointer registered = resample->GetOutput();

		printf("  Registration done! It took roughly %d seconds\n",(int)(time(NULL)-t_ini_step));
		std::cout << std::flush;


		/***********************************/
		/*  Step 3: bias field correction  */
		/***********************************/

		std::cout << "Step 3 of 9: rough bias field correction..."  << std::flush;
		t_ini_step = time(NULL);

		ShrinkerType::Pointer shrinker = ShrinkerType::New();
		shrinker->SetInput( registered );
		shrinker->SetShrinkFactors( SHRINK_FACTOR_BFC );

		MaskImageType::Pointer maskImage = readerMe->GetOutput(); 
		MaskShrinkerType::Pointer maskshrinker = MaskShrinkerType::New();
		maskshrinker->SetInput( maskImage );
		maskshrinker->SetShrinkFactors( SHRINK_FACTOR_BFC ); 

		shrinker->Update();
		maskshrinker->Update(); 


		CorrecterType::Pointer correcter = CorrecterType::New();
		correcter->SetInput( shrinker->GetOutput() ); 
		correcter->SetMaskImage( maskshrinker->GetOutput() ); 
		correcter->SetConvergenceThreshold(1e-9);
		itk::Array<unsigned int>   its(1); 
		its[0]=200; // default is 50
		correcter->SetMaximumNumberOfIterations(its); 
		typedef N4CommandIterationUpdate<CorrecterType> N4CommandType;
		N4CommandType::Pointer n4observer = N4CommandType::New();
		correcter->AddObserver( itk::IterationEvent(), n4observer );
		try
    		{
    			correcter->Update();
		}
  		catch( itk::ExceptionObject & err )
    		{
    			std::cout << "ExceptionObject caught is bias field correction!" << std::endl;
    			std::cout << err << std::endl << std::flush;
    			return EXIT_FAILURE;
    		}

		BSplinerType::Pointer bspliner = BSplinerType::New();
		bspliner->SetInput( correcter->GetLogBiasFieldControlPointLattice() );
		bspliner->SetSplineOrder( correcter->GetSplineOrder() );
		bspliner->SetSize( registered->GetLargestPossibleRegion().GetSize() );
		bspliner->SetOrigin( registered->GetOrigin() );
		bspliner->SetDirection( registered->GetDirection() );
		bspliner->SetSpacing( registered->GetSpacing() );
		bspliner->Update();

		ImageType::Pointer logField = ImageType::New();
		logField->SetOrigin( bspliner->GetOutput()->GetOrigin() );
		logField->SetSpacing( bspliner->GetOutput()->GetSpacing() );
		logField->SetRegions(    bspliner->GetOutput()->GetLargestPossibleRegion().GetSize() );
		logField->SetDirection( bspliner->GetOutput()->GetDirection() );
		logField->Allocate();

		itk::ImageRegionIterator<CorrecterType::ScalarImageType> ItB(bspliner->GetOutput(), bspliner->GetOutput()->GetLargestPossibleRegion() );
		itk::ImageRegionIterator<ImageType> ItF( logField,logField->GetLargestPossibleRegion() );
		for( ItB.GoToBegin(), ItF.GoToBegin(); !ItB.IsAtEnd(); ++ItB, ++ItF )
		{
			ItF.Set( ItB.Get()[0] );
		}


		ExpFilterType::Pointer expFilter = ExpFilterType::New();
		expFilter->SetInput( logField );
		expFilter->Update();

		DivideImageFilterType::Pointer divider = DivideImageFilterType::New();
		divider->SetInput1( registered );
		divider->SetInput2( expFilter->GetOutput() );
		divider->Update();

		ImageType::Pointer corrected=divider->GetOutput();


		// get robust min/max and equalize
		ImageType::SizeType size=corrected->GetBufferedRegion().GetSize();
		PixelType *data_p=corrected->GetBufferPointer();
		MaskPixelType *data_m=readerMd->GetOutput()->GetBufferPointer();
		double *data_in = (double *) malloc(size[0]*size[1]*size[2]*sizeof(double));
		if(data_in==NULL){
			fprintf(stderr,"\n Ran out of memory, exiting... \n");
			exit(1);
		}
		int nn=0;
		int i=0;
		while(i<size[0]*size[1]*size[2])
		{
			if((*(data_m+i))>0)
			{
				data_in[nn]=(double)(*(data_p+i));
				nn++;
			}
			i++;
		}

		vector<double> myvector (data_in, data_in+nn);
		sort (myvector.begin(), myvector.end()); 
		double robMin=myvector.at((int)(nn*0.01));
		double robMax=myvector.at((int)(nn*0.99));
		free(data_in);

		int auxx2;
		double h[1001];
		for(int j=0; j<1001; j++) h[j]=0;
		i=0;
		while(i<size[0]*size[1]*size[2])
		{
			auxx2=200.0+((double)(*(data_p+i))-robMin)/(robMax-robMin)*600.0;
			if(auxx2<0) auxx2=0;
			if(auxx2>1000) auxx2=1000;
			if((*(data_m+i))>0) h[auxx2]=h[auxx2]+1;
			*(data_p+i)=(double)auxx2;
			i++;
		}


		double c[1001]; c[0]=h[0]; for(int j=1; j<1001; j++) c[j]=c[j-1]+h[j];
		for(int j=0; j<1001; j++)
		{
			h[j]=h[j]/(double)nn;
			c[j]=c[j]/c[1000]*1000;
		}

		i=0;
		while(i<size[0]*size[1]*size[2])
		{
			*(data_p+i)=(double)c[(int)(*(data_p+i))];
			i++;
		}


		printf("Done! It took roughly %d seconds\n",(int)(time(NULL)-t_ini_step));
		std::cout << std::flush;




		/***********************************/
		/*   Step 4: Feature calculation   */
		/***********************************/

		std::cout << "Step 4 of 9: calculating features..."  << std::flush;
		t_ini_step = time(NULL);

		const int max_order = 2;
		const int no_scales = 3;
		float sigmas2[3]={0.5,2.0,8.0};
		// feature order in original Matlab version...
		int order[30]={1,2,4,7,3,5,6,8,9,10,12,13,15,18,14,16,17,19,20,21,23,24,26,29,25,27,28,30,31,32}; 
		int grad_order[3]={0,11,22};
		int grad_comps[3][3]={{2,4,7},{13,15,18},{24,26,29}};
		double means[]={81.0232,439.3682,-2.3058,0.035438,2.1901,-0.23174,-1.3498,-0.22016,-0.43238,0.048938,-2.6184,55.374,440.2726,-2.4151,1.2696,1.4408,-0.11173,0.80683,-0.21245,-0.13055,0.019784,1.4639,29.7145,450.6124,-2.0719,1.2689,0.17289,0.029311,1.2964,-0.17644,0.024883,-0.0017843,2.615,65.2635,57.3589,90.6642};
		double stds[]={40.7004,154.2286,43.5293,65.214,43.2907,13.3516,60.308,51.0131,26.4112,14.0375,67.9257,26.3091,130.8698,29.1458,20.6909,28.8305,6.2023,19.0772,34.9552,6.5288,6.837,23.6271,13.1709,102.5213,16.3552,5.1685,15.442,2.1299,5.2682,17.8113,1.0467,2.5996,7.7468,27.1293,23.3043,22.4309};
		//int fv[36]={7,1,15,-1,-1,-1,-1,-1,-1,-1,-1,4,0,9,-1,16,-1,-1,-1,-1,-1,14,13,3,5,11,-1,-1,12,-1,-1,-1,6,8,10,2};


		GaussianFilterType::Pointer filter = GaussianFilterType::New();
		DerivativeFilterType::Pointer filterX = DerivativeFilterType::New();
		DerivativeFilterType::Pointer filterY = DerivativeFilterType::New();
		DerivativeFilterType::Pointer filterZ = DerivativeFilterType::New();
		ShiftScaleFilterType::Pointer shiftScale = ShiftScaleFilterType::New();
		filterX->SetDirection(0);
		filterY->SetDirection(1);
		filterZ->SetDirection(2);
		filter->SetInput(corrected);
		filter->SetUseImageSpacingOn();


		int feat_n=0;
		for(int i=0; i<no_scales; i++){
			for(int j=0; j<=max_order; j++){
				// order j
				for(int ox=0; ox<=j; ox++){
					for(int oy=0; oy<=j; oy++){
						for(int oz=0; oz<=j; oz++){
							if(ox+oy+oz==j){
								filter->SetVariance(sigmas2[i]);
								filter->SetMaximumKernelWidth( 6*sigmas2[i] );
								filter->Update();
								if(ox>0){
									filterX->SetInput(filter->GetOutput());
									filterX->SetOrder(ox);
									filterX->Update();
								}
								if(oy>0){
									if(ox>0){
										filterY->SetInput(filterX->GetOutput());
									}
									else{
										filterY->SetInput(filter->GetOutput());
									}
									filterY->SetOrder(oy);
									filterY->Update();
								}
								if(oz>0){
									if(oy>0){
										filterZ->SetInput(filterY->GetOutput());
									}
									else if(ox>0){
										filterZ->SetInput(filterY->GetOutput());
									}
									else{
										filterZ->SetInput(filter->GetOutput());
									}
									filterZ->SetOrder(oz);
									filterZ->Update();
								}

								if(oz>0){
									shiftScale->SetInput(filterZ->GetOutput());
								} else if(oy>0){
									shiftScale->SetInput(filterY->GetOutput());
								} else if(ox>0){
									shiftScale->SetInput(filterX->GetOutput());
								}else{
									shiftScale->SetInput(filter->GetOutput());
								}

								int found=0;
								for(int ii=0; ii<3; ii++){
									for(int jj=0; jj<3; jj++){
										if(grad_comps[ii][jj]==order[feat_n])
											found=1;
									}
								}
								if(found==1){
									writer->SetInput(shiftScale->GetInput());
									sprintf(filename,"%s%s%d_noNorm.nii.gz\0",temp_dir,PATH_SEPARATOR,order[feat_n]); 
									writer->SetFileName(filename); 
									writer->Update();
								}

								shiftScale->SetScale(1.0/stds[order[feat_n]]);
								shiftScale->SetShift(-means[order[feat_n]]);
								shiftScale->Update();
								writer->SetInput(shiftScale->GetOutput());
								sprintf(filename,"%s%s%d.nii.gz\0",temp_dir,PATH_SEPARATOR,order[feat_n]);
								writer->SetFileName(filename);
								writer->Update();

								feat_n++;
							}
						}
					}
				}
			}
		}


		ReaderType::Pointer reader1 = ReaderType::New();
		ReaderType::Pointer reader2 = ReaderType::New();
		ReaderType::Pointer reader3 = ReaderType::New();
		AddFilterType::Pointer addFilt1 = AddFilterType::New();
		AddFilterType::Pointer addFilt2 = AddFilterType::New();
		SquareFilterType::Pointer powerFilt1 = SquareFilterType::New();
		SquareFilterType::Pointer powerFilt2 = SquareFilterType::New();
		SquareFilterType::Pointer powerFilt3 = SquareFilterType::New();
		SqrtFilterType::Pointer  sqrtFilt = SqrtFilterType::New();
		char filename1[1000], filename2[1000], filename3[1000], outputfilename[1000];


		// gradients
		for(int s=0; s<no_scales; s++){
			sprintf(filename1,"%s%s%d_noNorm.nii.gz\0",temp_dir,PATH_SEPARATOR,grad_comps[s][0]);
			sprintf(filename2,"%s%s%d_noNorm.nii.gz\0",temp_dir,PATH_SEPARATOR,grad_comps[s][1]);
			sprintf(filename3,"%s%s%d_noNorm.nii.gz\0",temp_dir,PATH_SEPARATOR,grad_comps[s][2]);
			sprintf(outputfilename,"%s%s%d.nii.gz\0",temp_dir,PATH_SEPARATOR,grad_order[s]);
			reader1->SetFileName(filename1); reader1->Update();
			reader2->SetFileName(filename2); reader2->Update();
			reader3->SetFileName(filename3); reader3->Update();

			ImageType::Pointer outputImage = ImageType::New();
			outputImage->SetRegions( reader1->GetOutput()->GetRequestedRegion() );
			outputImage->CopyInformation( reader1->GetOutput() );
			outputImage->Allocate();
			IteratorType outputIt( outputImage, outputImage->GetRequestedRegion() );
			IteratorType inputIt1( reader1->GetOutput(), reader1->GetOutput()->GetRequestedRegion() );
			IteratorType inputIt2( reader2->GetOutput(), reader2->GetOutput()->GetRequestedRegion() );
			IteratorType inputIt3( reader3->GetOutput(), reader3->GetOutput()->GetRequestedRegion() );
			ImageType::IndexType requestedIndex =outputImage->GetRequestedRegion().GetIndex();
			ImageType::SizeType requestedSize = outputImage->GetRequestedRegion().GetSize();
			PixelType aux1, aux2, aux3;	
			for ( outputIt.GoToBegin(),inputIt1.GoToBegin(),inputIt2.GoToBegin(),inputIt3.GoToBegin() ; !outputIt.IsAtEnd(); ++outputIt,++inputIt1,++inputIt2,++inputIt3)
			{
				aux1=inputIt1.Get();
				aux2=inputIt2.Get();
				aux3=inputIt3.Get();
				outputIt.Set(sqrt(aux1*aux1+aux2*aux2+aux3*aux3));
			}
			shiftScale->SetInput(outputImage); shiftScale->SetScale(1.0/stds[grad_order[s]]); shiftScale->SetShift(-means[grad_order[s]]); 
			shiftScale->Update(); writer->SetInput(shiftScale->GetOutput());  writer->SetFileName(outputfilename); writer->Update();
		}

		// location features
		ImageType::Pointer outputImageX = ImageType::New();
		ImageType::Pointer outputImageY = ImageType::New();
		ImageType::Pointer outputImageZ = ImageType::New();
		outputImageX->SetRegions( corrected->GetRequestedRegion() );
		outputImageY->SetRegions( corrected->GetRequestedRegion() );
		outputImageZ->SetRegions( corrected->GetRequestedRegion() );
		outputImageX->CopyInformation( corrected );
		outputImageY->CopyInformation( corrected );
		outputImageZ->CopyInformation( corrected );
		outputImageX->Allocate();
		outputImageY->Allocate();
		outputImageZ->Allocate();

		IteratorType outputItX( outputImageX, outputImageX->GetRequestedRegion() );
		IteratorType outputItY( outputImageY, outputImageY->GetRequestedRegion() );
		IteratorType outputItZ( outputImageZ, outputImageZ->GetRequestedRegion() );
		ImageType::IndexType requestedIndexX =outputImageX->GetRequestedRegion().GetIndex();
		ImageType::IndexType requestedIndexY =outputImageY->GetRequestedRegion().GetIndex();
		ImageType::IndexType requestedIndexZ =outputImageZ->GetRequestedRegion().GetIndex();
		ImageType::SizeType requestedSize = outputImageX->GetRequestedRegion().GetSize();
		for ( outputItX.GoToBegin(),outputItY.GoToBegin(),outputItZ.GoToBegin() ; !outputItX.IsAtEnd(); ++outputItX,++outputItY,++outputItZ)
		{
			ImageType::IndexType idx = outputItX.GetIndex();
			outputItX.Set(idx[0]+1);
			outputItY.Set(idx[1]+1);
			outputItZ.Set(idx[2]+1);
		}

		char filenameX[1000],filenameY[1000],filenameZ[1000];
		sprintf(filenameX,"%s%s34.nii.gz\0",temp_dir,PATH_SEPARATOR);
		sprintf(filenameY,"%s%s33.nii.gz\0",temp_dir,PATH_SEPARATOR);
		sprintf(filenameZ,"%s%s35.nii.gz\0",temp_dir,PATH_SEPARATOR);

		shiftScale->SetInput(outputImageX); shiftScale->SetScale(1.0/stds[34]); shiftScale->SetShift(-means[34]); shiftScale->Update();
		writer->SetInput(shiftScale->GetOutput());  writer->SetFileName(filenameX); writer->Update();
		shiftScale->SetInput(outputImageY); shiftScale->SetScale(1.0/stds[33]); shiftScale->SetShift(-means[33]); shiftScale->Update();
		writer->SetInput(shiftScale->GetOutput());  writer->SetFileName(filenameY); writer->Update();
		shiftScale->SetInput(outputImageZ); shiftScale->SetScale(1.0/stds[35]); shiftScale->SetShift(-means[35]); shiftScale->Update();
		writer->SetInput(shiftScale->GetOutput());  writer->SetFileName(filenameZ); writer->Update();




		printf("Done! It took roughly %d seconds\n",(int)(time(NULL)-t_ini_step));
		std::cout << std::flush;








		/***********************************/
		/*   Step 5: Pixel Classification  */
		/***********************************/

		std::cout << "Step 5 of 9: voxel classification..."  << std::flush;
		t_ini_step = time(NULL);

		// Selected features
		int fv[10]={12,    35,    24,    23,    11,    33,    26,     1,    32,    34};

		ImageType::Pointer features[10];
		ReaderType::Pointer featureReader[10];
		for(int i=0;i<10; i++){
			featureReader[i]=ReaderType::New();
			sprintf(filename,"%s%s%d.nii.gz\0",temp_dir,PATH_SEPARATOR,fv[i]);
			featureReader[i]->SetFileName(filename);
			featureReader[i]->Update();
			features[i]=featureReader[i]->GetOutput();
		}

		IteratorType inputIts[10]={IteratorType(features[0],features[0]->GetRequestedRegion()),IteratorType(features[1],features[1]->GetRequestedRegion()),IteratorType(features[2],features[2]->GetRequestedRegion()),IteratorType(features[3],features[3]->GetRequestedRegion()),IteratorType(features[4],features[4]->GetRequestedRegion()),IteratorType(features[5],features[5]->GetRequestedRegion()),IteratorType(features[6],features[6]->GetRequestedRegion()),IteratorType(features[7],features[7]->GetRequestedRegion()),IteratorType(features[8],features[8]->GetRequestedRegion()),IteratorType(features[9],features[9]->GetRequestedRegion())}; 
		for (int i=0; i<10; i++){
			inputIts[i].GoToBegin();
		}
		MaskIteratorType maskIt( readerMini->GetOutput(), readerMini->GetOutput()->GetRequestedRegion() );
		maskIt.GoToBegin();
		double *X=(double *)malloc(10*MINI_PIXS*sizeof(double));
		if(X==NULL){
			fprintf(stderr,"\n Ran out of memory, exiting... \n");
			exit(1);
		}
		double *pos=X;
		while(!maskIt.IsAtEnd())
		{
			if(maskIt.Get()>0){
				for (int i=0; i<10; i++){
					*pos=(double)inputIts[i].Get();
					pos++;
					++inputIts[i];
				}					
			}
			else{
				for (int i=0; i<10; i++) 
					++inputIts[i];
			}
			++maskIt;

		}


		// Load forest
		int *nodestatus=load_nodestatus();
		int *bestvar=load_bestvar();
		double *xbestsplit=load_xbestsplit();
		int *nodeclass=load_nodeclass();
		int *treemap=load_treemap();
		double *votes=(double *)malloc(MINI_PIXS*sizeof(double));
		int ntrees=load_ntree();
		int nrnodes=load_nrnodes();
		for(int vi=0; vi<MINI_PIXS; vi++)  // Go along data points
			votes[vi]=0;
				
	
		// These guys store a single tree at the time
		int *nodestatus_t=(int *)malloc(nrnodes*sizeof(int));
		int *bestvar_t=(int *)malloc(nrnodes*sizeof(int));
		double *xbestsplit_t=(double *)malloc(nrnodes*sizeof(double));
		int *nodeclass_t=(int *)malloc(nrnodes*sizeof(int));
		int *treemap_t=(int *)malloc(2*nrnodes*sizeof(int));
		for(int u=0; u<2*nrnodes; u++)
			treemap_t[u]=0;
		for(int u=0; u<nrnodes; u++){ 
			nodestatus_t[u]=0;
			xbestsplit_t[u]=0;
			nodeclass_t[u]=0;
			bestvar_t[u]=0;
		}

		// Testing!
		for(int t=0; t<ntrees; t++){  // go along trees

			// Copy current tree to _t variables
			 for(int u=0; u<MAX_R; u++){
				 nodestatus_t[u]=nodestatus[MAX_R*t+u];
				 xbestsplit_t[u]=xbestsplit[MAX_R*t+u];
				 nodeclass_t[u]=nodeclass[MAX_R*t+u];
				 bestvar_t[u]=bestvar[MAX_R*t+u];
			 }
			 if(2*MAX_R<nrnodes){
				 for(int u=0; u<(2*MAX_R); u++){
				 treemap_t[u]=treemap[4*MAX_R*t+u];
				 treemap_t[u+nrnodes]=treemap[4*MAX_R*t+u+2*MAX_R];
			 }
			 }
			 else{
				 for(int u=0; u<nrnodes; u++){
					 treemap_t[u]=treemap[2*nrnodes*t+u];
					 treemap_t[u+nrnodes]=treemap[2*nrnodes*t+u+nrnodes];
				 }
			 }
			
			double *x=X;
			for(int vi=0; vi<MINI_PIXS; vi++){  // Go along data points
				int k=0;     // indices the node
				int m=0;     // indices the best variable
		
				while (nodestatus_t[k] != NODE_TERMINAL) {
					m = bestvar_t[k] - 1;
					k = (x[m] <= xbestsplit_t[k]) ?
		                   treemap_t[k * 2] - 1 : treemap_t[1 + k * 2] - 1;
				}
			        /* Leaf node */
				if(nodeclass_t[k]>1){
					votes[vi]=votes[vi]+1;
				}
				x=x+10;
		        }
		}
		
		
		// Free memory!
		free(nodestatus);
		free(bestvar);
		free(xbestsplit);
		free(nodeclass);
		free(treemap);
		free(X);
                free(nodestatus_t);
		free(bestvar_t);
		free(xbestsplit_t);
		free(nodeclass_t);
		free(treemap_t);


                // Build vote image
		ImageType::Pointer votesI = ImageType::New();
		votesI->SetRegions( features[0]->GetRequestedRegion() );
		votesI->CopyInformation( features[0] );
		votesI->Allocate();
		IteratorType voteIt( votesI, votesI->GetRequestedRegion() );
		voteIt.GoToBegin();
		// double *v=votes+1;
		double *v=votes;
		maskIt.GoToBegin();
		while(!voteIt.IsAtEnd()){
			if(maskIt.Get()>0){
				voteIt.Set((float)(*v));
				// v+=2;
				v++;
			}
			else{
				voteIt.Set(0.0f);
			}
			++voteIt;
			++maskIt;
		}
		free(votes); 


		GaussianFilterType::Pointer filterR = GaussianFilterType::New();
		filterR->SetInput(votesI);
		filterR->SetUseImageSpacingOn();
		filterR->SetVariance(SIGMA_R*PIXSIZE);
		filterR->SetMaximumKernelWidth( 6*0.5 );
		filterR->Update();
		shiftScale->SetInput(filterR->GetOutput()); shiftScale->SetScale(1.0/200); shiftScale->SetShift(0); shiftScale->Update();
		ImageType::Pointer R=shiftScale->GetOutput();	


		printf("Done! It took roughly %d seconds\n",(int)(time(NULL)-t_ini_step));
		std::cout << std::flush;





		/***********************************/
		/*   Step 6: Fitting shape model   */
		/***********************************/

		std::cout << "Step 6 of 9: fitting the shape model..."  << std::flush;
		t_ini_step = time(NULL);


		double rot21_v[361]={0.39263, -0.39823, -0.056966, -0.21999, 0.09444, 0.24357, 0.15485, -0.46521, 0.17349, -0.070854, -0.48093, -0.0066558, 0.12239, -0.18529, -0.015539, -0.020789, 0.040593, 0.084622, 0.041529, 0.27627, 0.39515, 0.41699, 0.058066, -0.076354, -0.19648, -0.078738, -0.015821, -0.35603, 0.032815, -0.23293, -0.025533, 0.26792, -0.43686, 0.12641, -0.061764, -0.19286, -0.00019414, -0.19247, -0.09979, -0.05159, 0.50539, -0.22027, 0.56295, 0.18065, 0.045703, 0.17977, -0.011414, 0.31589, -0.080243, -0.047593, -0.20028, 0.02875, -0.057481, 0.037977, 0.30607, -0.13283, -0.18402, 0.062336, -0.043772, 0.32161, 0.5032, 0.34665, 0.13953, -0.0020872, -0.24248, -0.14573, -0.26132, 0.20113, -0.083682, 0.036608, 0.10517, -0.0847, -0.17968, -0.092826, 0.050441, 0.48899, 0.47364, -0.0019273, -0.28482, 0.045449, 0.46201, -0.11204, -0.3133, 0.096352, 0.041434, 0.0055339, 0.25486, -0.087825, 0.21417, 0.17239, -0.051339, 0.10493, -0.070429, 0.27496, -0.33884, -0.0014526, 0.072082, -0.12222, -0.14327, -0.097442, 0.46388, -0.30035, 0.060679, -0.28522, -0.048605, 0.19397, 0.12162, -0.10763, -0.3187, 0.17997, -0.12721, 0.39684, 0.41242, 0.12385, -0.1508, -0.11973, 0.066719, -0.098201, 0.14266, -0.31325, 0.39493, 0.082904, -0.10943, -0.42705, 0.11553, 0.21618, -0.023151, -0.2888, -0.30119, 0.3188, 0.095179, 0.35161, -0.064013, 0.1657, -0.087661, 0.22572, 0.31494, -0.17465, 0.17923, -0.1958, 0.46792, 0.3031, -0.31774, -0.27349, 0.1605, 0.14507, 0.088015, 0.11014, 0.33609, 0.20983, -0.055415, 0.021168, -0.30889, -0.28131, 0.18572, 0.12505, 0.00097464, 0.017817, -0.024227, 0.04255, 0.50103, 0.091189, 0.24513, -0.061666, 0.26313, -0.35747, 0.19517, -0.30862, -0.17577, 0.20783, -0.21196, -0.060977, -0.071535, 0.0015878, 0.19668, -0.072329, -0.13795, -0.20758, 0.068828, 0.04079, 0.58877, -0.24522, 0.070266, -0.0049977, -0.081048, -0.40331, 0.18064, -0.076948, 0.39567, 0.32039, 0.52501, -0.1806, 0.057768, 0.021153, -0.1218, -0.38543, 0.070533, 0.1702, 0.16341, 0.11049, 0.254, 0.036046, -0.38241, -0.24044, 0.0394, -0.23871, 0.22576, -0.17108, 0.20755, 0.17237, 0.028366, 0.19518, -0.077064, -0.099756, 0.20933, 0.074224, -0.18417, 0.044071, 0.19546, 0.32226, 0.7721, 0.084999, 0.13803, -0.03957, 0.099719, -0.20818, -0.07898, -0.068983, -0.033755, -0.40729, -0.10078, 0.15346, -0.075972, -0.094143, 0.1962, 0.12064, -0.4284, 0.2135, 0.054152, 0.052333, 0.55674, 0.10244, 0.020426, -0.12683, 0.36192, -0.16721, -0.030137, 0.2105, 0.15123, 0.17131, -0.096031, -0.27351, 0.22406, 0.34768, 0.25579, 0.046705, -0.042545, -0.018207, -0.17003, 0.042709, 0.32289, -0.36894, -0.42206, -0.02768, 0.33397, -0.13063, 0.074635, 0.037224, -0.12599, 0.26947, 0.12133, 0.055697, 0.50382, 0.13056, -0.080718, 0.21983, -0.12287, 0.038778, -0.2079, 0.092346, 0.60048, 0.12186, -0.1435, 0.31088, 0.00014503, 0.14454, 0.081836, 0.14683, -0.14037, -0.20568, 0.18448, 0.19407, -0.10551, 0.097224, 0.16888, 0.40544, -0.4761, 0.21039, -0.082728, 0.025615, 0.54153, 0.050806, -0.033953, 0.16805, -0.043436, 0.55702, -0.14643, 0.18288, 0.12427, -0.11741, 0.18055, -0.2584, 0.37513, 0.075185, -0.050505, 0.13146, 0.21384, -0.022151, -0.030425, -0.092783, 0.53065, 0.011854, 0.025697, -0.053276, -0.079294, 0.34942, -0.313, -0.1292, -0.4123, -0.20484, -0.22248, 0.023029, -0.061174, -0.020124, 0.0034859, 0.071623, 0.44188, 0.34805, -0.0051317, 0.16785, 0.34952, 0.12687, 0.0047319, 0.14967, -0.12713, -0.44483, 0.27317, -0.016597, 0.07391, 0.39946, 0.10295, -0.033151, -0.05049, 0.10839, 0.34628, -0.058346, 0.10832, -0.12033, -0.21798, -0.073525, 0.5433};		
		vnl_matrix<double> ROT21(rot21_v,N_MODES,N_MODES);
		double rot12_v[361]={0.39263, 0.27627, -0.09979, 0.062336, 0.47364, -0.0014526, -0.1508, 0.1657, -0.30889, -0.060977, 0.52501, 0.17237, -0.033755, 0.2105, 0.074635, 0.14454, -0.043436, -0.053276, 0.0047319, -0.39823, 0.39515, -0.05159, -0.043772, -0.0019273, 0.072082, -0.11973, -0.087661, -0.28131, -0.071535, -0.1806, 0.028366, -0.40729, 0.15123, 0.037224, 0.081836, 0.55702, -0.079294, 0.14967, -0.056966, 0.41699, 0.50539, 0.32161, -0.28482, -0.12222, 0.066719, 0.22572, 0.18572, 0.0015878, 0.057768, 0.19518, -0.10078, 0.17131, -0.12599, 0.14683, -0.14643, 0.34942, -0.12713, -0.21999, 0.058066, -0.22027, 0.5032, 0.045449, -0.14327, -0.098201, 0.31494, 0.12505, 0.19668, 0.021153, -0.077064, 0.15346, -0.096031, 0.26947, -0.14037, 0.18288, -0.313, -0.44483, 0.09444, -0.076354, 0.56295, 0.34665, 0.46201, -0.097442, 0.14266, -0.17465, 0.00097464, -0.072329, -0.1218, -0.099756, -0.075972, -0.27351, 0.12133, -0.20568, 0.12427, -0.1292, 0.27317, 0.24357, -0.19648, 0.18065, 0.13953, -0.11204, 0.46388, -0.31325, 0.17923, 0.017817, -0.13795, -0.38543, 0.20933, -0.094143, 0.22406, 0.055697, 0.18448, -0.11741, -0.4123, -0.016597, 0.15485, -0.078738, 0.045703, -0.0020872, -0.3133, -0.30035, 0.39493, -0.1958, -0.024227, -0.20758, 0.070533, 0.074224, 0.1962, 0.34768, 0.50382, 0.19407, 0.18055, -0.20484, 0.07391, -0.46521, -0.015821, 0.17977, -0.24248, 0.096352, 0.060679, 0.082904, 0.46792, 0.04255, 0.068828, 0.1702, -0.18417, 0.12064, 0.25579, 0.13056, -0.10551, -0.2584, -0.22248, 0.39946, 0.17349, -0.35603, -0.011414, -0.14573, 0.041434, -0.28522, -0.10943, 0.3031, 0.50103, 0.04079, 0.16341, 0.044071, -0.4284, 0.046705, -0.080718, 0.097224, 0.37513, 0.023029, 0.10295, -0.070854, 0.032815, 0.31589, -0.26132, 0.0055339, -0.048605, -0.42705, -0.31774, 0.091189, 0.58877, 0.11049, 0.19546, 0.2135, -0.042545, 0.21983, 0.16888, 0.075185, -0.061174, -0.033151, -0.48093, -0.23293, -0.080243, 0.20113, 0.25486, 0.19397, 0.11553, -0.27349, 0.24513, -0.24522, 0.254, 0.32226, 0.054152, -0.018207, -0.12287, 0.40544, -0.050505, -0.020124, -0.05049, -0.0066558, -0.025533, -0.047593, -0.083682, -0.087825, 0.12162, 0.21618, 0.1605, -0.061666, 0.070266, 0.036046, 0.7721, 0.052333, -0.17003, 0.038778, -0.4761, 0.13146, 0.0034859, 0.10839, 0.12239, 0.26792, -0.20028, 0.036608, 0.21417, -0.10763, -0.023151, 0.14507, 0.26313, -0.0049977, -0.38241, 0.084999, 0.55674, 0.042709, -0.2079, 0.21039, 0.21384, 0.071623, 0.34628, -0.18529, -0.43686, 0.02875, 0.10517, 0.17239, -0.3187, -0.2888, 0.088015, -0.35747, -0.081048, -0.24044, 0.13803, 0.10244, 0.32289, 0.092346, -0.082728, -0.022151, 0.44188, -0.058346, -0.015539, 0.12641, -0.057481, -0.0847, -0.051339, 0.17997, -0.30119, 0.11014, 0.19517, -0.40331, 0.0394, -0.03957, 0.020426, -0.36894, 0.60048, 0.025615, -0.030425, 0.34805, 0.10832, -0.020789, -0.061764, 0.037977, -0.17968, 0.10493, -0.12721, 0.3188, 0.33609, -0.30862, 0.18064, -0.23871, 0.099719, -0.12683, -0.42206, 0.12186, 0.54153, -0.092783, -0.0051317, -0.12033, 0.040593, -0.19286, 0.30607, -0.092826, -0.070429, 0.39684, 0.095179, 0.20983, -0.17577, -0.076948, 0.22576, -0.20818, 0.36192, -0.02768, -0.1435, 0.050806, 0.53065, 0.16785, -0.21798, 0.084622, -0.00019414, -0.13283, 0.050441, 0.27496, 0.41242, 0.35161, -0.055415, 0.20783, 0.39567, -0.17108, -0.07898, -0.16721, 0.33397, 0.31088, -0.033953, 0.011854, 0.34952, -0.073525, 0.041529, -0.19247, -0.18402, 0.48899, -0.33884, 0.12385, -0.064013, 0.021168, -0.21196, 0.32039, 0.20755, -0.068983, -0.030137, -0.13063, 0.00014503, 0.16805, 0.025697, 0.12687, 0.5433};
		vnl_matrix<double> ROT12(rot12_v,N_MODES,N_MODES);


		IteratorType RIt( R, R->GetRequestedRegion() );
		RIt.GoToBegin();
		ImageType::Pointer C = ImageType::New();
		C->SetRegions( R->GetRequestedRegion() );
		C->CopyInformation( R );
		C->Allocate();
		IteratorType CIt( C, C->GetRequestedRegion() );
		CIt.GoToBegin();

		while(!RIt.IsAtEnd())
		{
			CIt.Set(log(EPS+(1-EPS)*(1-RIt.Get())));
			++RIt; ++CIt;
		}

		double *lambda_v=load_lambda();  vnl_vector<double> lambda(lambda_v,N_MODES);
		double *lambda2_v=load_lambda2(); vnl_vector<double> lambda2(lambda2_v,N_MODES);
		double *mean_normals_v=load_mean_normals();   vnl_vector<double> mean_normals(mean_normals_v,3*NL);
		double *mu_v = load_mu(); vnl_vector<double> mu(mu_v,3*NL);
		double *PHI_v = load_PHI(); vnl_matrix<double> PHI(PHI_v,NL*3,N_MODES);
		double *PHI2_v = load_PHI2(); vnl_matrix<double> PHI2(PHI2_v,NL*3,N_MODES);
		double *face_v = load_face(); vnl_matrix<double> face(face_v,N_FACES,3);

		vnl_vector<double> best_contour=mu;
		vnl_vector<double> contour=mu;
		vnl_vector<double> contour_old(NL*3,0.0);
		vnl_vector<double> cont=mu;
		vnl_vector<double> b_aux(N_MODES,0.0);
		vnl_vector<double> b(N_MODES,0.0);
		b=b_aux;
		bool ready=false;
		int it=0;
		double curr_cost=0;
		double best_cost=1e10;
		bool calc_best_contour=false;

		LinearInterpolatorType::Pointer Cinterp = LinearInterpolatorType::New();
		Cinterp->SetInputImage(C);
		LinearInterpolatorType::ContinuousIndexType position;
		ImageType::RegionType::SizeType imSize=C->GetLargestPossibleRegion().GetSize();

		while(ready==false)
		{
			int *order=randperm(N_MODES);

			for(int index=0; index<N_MODES; index++)
			{
				int m=*(order+index);
				vnl_vector<double> b_back = b;
				if(it<MAX_IT_ALL){
					int lc=1+floor(6*sqrt(lambda2[m])/PREC_B);
					double *cand=(double *) malloc(lc*sizeof(double));     
					double *cost=(double *) malloc(lc*sizeof(double));
					if(cand==NULL || cost==NULL){
						fprintf(stderr,"\n Ran out of memory, exiting... \n");
						exit(1);
					}
					for(int c=0; c<lc; c++){
						cand[c]=-3*sqrt(lambda2[m])+(double)c*PREC_B;
						b=b_back;
						b[m]=cand[c];
						cont=mu+PHI2*b;
						cost[c]=0;
						for(int l=0; l<NL; l++){
							position[1]=cont[l]-1;	position[0]=cont[l+NL]-1;	position[2]=cont[l+2*NL]-1;						
							if((position[0]<0 || position[1]<0 || position[2]<0 || position[0]>imSize[0]-1  || position[1]>imSize[1]-1  || position[2]>imSize[2]-1)==false)
								cost[c]+=Cinterp->EvaluateAtContinuousIndex(position);
						}
					}
					curr_cost=1e20;
					int pos=-1;
					for(int c=0; c<lc; c++){
						if(cost[c]<=curr_cost){
							curr_cost=cost[c];
							pos=c;
						}
					}
					curr_cost=curr_cost/(double)NL;  // cost defined as mean
					b[m]=cand[pos];
					free(cand);
					free(cost);
				}
				else
				{
					int done=0;
					int iter=0;
					double step=STEP_INI;
					while(done==0){
						iter++;
						b_back=b;
						vnl_vector<double> b_l(3,0.0); b_l=b_back; b_l[m]=b_back[m]-DELTA_B;
						vnl_vector<double> b_r(3,0.0); b_r=b_back; b_r[m]=b_back[m]+DELTA_B;

						cont=mu+PHI2*b;
						vnl_vector<double> cont_l=mu+PHI2*b_l;
						vnl_vector<double> cont_r=mu+PHI2*b_r;


						double cost, cost_l, cost_r;
						double cc, cl, cr;
						bool ok=true;
						for(int l=0; l<NL; l++){
							ok=true;
							position[1]=cont[l]-1; position[0]=cont[l+NL]-1; position[2]=cont[l+2*NL]-1;						
							if((position[0]<0 || position[1]<0 || position[2]<0 || position[0]>imSize[0]-1  || position[1]>imSize[1]-1  || position[2]>imSize[2]-1)==false){
								cc=Cinterp->EvaluateAtContinuousIndex(position);
							}
							else{
								ok=false;
							}
							position[1]=cont_l[l]-1; position[0]=cont_l[l+NL]-1; position[2]=cont_l[l+2*NL]-1;						
							if((position[0]<0 || position[1]<0 || position[2]<0 || position[0]>imSize[0]-1  || position[1]>imSize[1]-1  || position[2]>imSize[2]-1)==false){
								cl=Cinterp->EvaluateAtContinuousIndex(position);
							}
							else{
								ok=false;
							}
							position[1]=cont_r[l]-1; position[0]=cont_r[l+NL]-1; position[2]=cont_r[l+2*NL]-1;	
							if((position[0]<0 || position[1]<0 || position[2]<0 || position[0]>imSize[0]-1  || position[1]>imSize[1]-1  || position[2]>imSize[2]-1)==false){
								cr=Cinterp->EvaluateAtContinuousIndex(position);
							}
							else{
								ok=false;
							}

							if(ok==true){
								cost+=cc;
								cost_l+=cl;
								cost_r+=cr;
							}
						}

						cost/=(double)NL; cost_r/=(double)NL; cost_l/=(double)NL;
						if(cost>curr_cost)
							step/=5.0;

						double fp=(cost_r-cost_l)/(2*DELTA_B);
						double fpp=(cost_r+cost_l-2*cost)/(DELTA_B*DELTA_B);

						while(fpp==0){
							fpp=0.002*((double)rand()/RAND_MAX-0.5);
						}

						double inc=-step*fp/fpp;

						vnl_vector<double> inc_v(N_MODES,0.0);
						inc_v[m]=inc;
						b=b+inc_v;
						if(b[m]>3*sqrt(lambda2[m]))
							b[m]=3*sqrt(lambda2[m]);
						if(b[m]<-3*sqrt(lambda2[m]))
							b[m]=-3*sqrt(lambda2[m]);

						if(abs(inc)<PREC_B_FINE || iter>MAX_IT_IN)
							done=1;

						curr_cost=cost;
					}
				}
			}
			free(order);



			if(curr_cost<=best_cost){
				best_cost=curr_cost;
				calc_best_contour=true;
			}
			else
			{
				calc_best_contour=false;
			}

			vnl_vector<double> borig = ROT21*b;
			double normb2=0;
			for(int m=0; m<N_MODES; m++)
				normb2+=borig[m]*borig[m]/lambda[m];
			if(normb2>NB2_MAX)
				borig=borig*sqrt(NB2_MAX/normb2);
			b=ROT12*borig;
			contour_old=contour;
			contour=mu+PHI2*b;
			if(calc_best_contour==true)
				best_contour=contour;


			vnl_vector<double> diff=contour-contour_old;
			double delta=diff.squared_magnitude();

			it++;

			if ((delta<DELTA && it>MAX_IT_ALL)  || it>=MAX_IT)
				ready=true;

		}
		contour=best_contour;



		printf("Done! It took roughly %d seconds\n",(int)(time(NULL)-t_ini_step));
		std::cout << std::flush;








		/************************************/
		/* Step 7: Sonka's free deformation */
		/************************************/

		std::cout << "Step 7 of 9: free deformation..."  << std::flush;
		t_ini_step = time(NULL);

		//update normals
		double *normals_v=(double *)malloc(3*NL*sizeof(double));
		if(normals_v==NULL){
			fprintf(stderr,"\n Ran out of memory, exiting... \n");
			exit(1);
		}
		vnl_vector<double> normals(normals_v,3*NL);
		double n_old_v[3]; vnl_vector<double> n_old(n_old_v,3);
		double d1_v[3]; vnl_vector<double> d1(d1_v,3);
		double d2_v[3]; vnl_vector<double> d2(d2_v,3);
		double n_new_v[3]; vnl_vector<double> n_new(n_new_v,3);

		double f_list[20];
		int n_found=0;
		int p1,pp1;
		for(int i=0; i<NL; i++)
		{
			n_found=0;
			for(int f=0; f<N_FACES; f++){
				if(face(f,0)==i || face(f,1)==i || face(f,2)==i){
					f_list[n_found]=f;
					n_found++;
				}
			}
			n_new[0]=0; n_new[1]=0; n_new[2]=0;
			n_old[0]=mean_normals[i]; n_old[1]=mean_normals[i+NL]; n_old[2]=mean_normals[i+2*NL];
			for(int f=0; f<n_found; f++)
			{
				if(face(f_list[f],0)==i){
					p1=face(f_list[f],1); pp1=face(f_list[f],2);
				}else if(face(f_list[f],1)==i){
					p1=face(f_list[f],0); pp1=face(f_list[f],2);
				}else{
					p1=face(f_list[f],0); pp1=face(f_list[f],1);
				}
				d1[0]=contour[p1]-contour[i]; d1[1]=contour[p1+NL]-contour[i+NL]; d1[2]=contour[p1+2*NL]-contour[i+2*NL];
				d2[0]=contour[pp1]-contour[i]; d2[1]=contour[pp1+NL]-contour[i+NL]; d2[2]=contour[pp1+2*NL]-contour[i+2*NL];

				double aux_v[3];
				cross_3d(aux_v,d1,d2);
				vnl_vector<double> aux(aux_v,3);

				aux=aux.normalize();
				if(dot_product(aux,n_old)<0)
					aux=aux*(-1.0);
				n_new=n_new+aux;			
			}
			n_new=n_new.normalize();
			normals[i]=n_new[0]; normals[i+NL]=n_new[1]; normals[i+2*NL]=n_new[2];
		}


		// let's sample!
		int SL=SEARCH_DIST/INC;
		int PL=1+2*SL;
		double *XI=(double *)malloc(NL*PL*sizeof(double));
		double *YI=(double *)malloc(NL*PL*sizeof(double));
		double *ZI=(double *)malloc(NL*PL*sizeof(double));
		double *COSTS=(double *)malloc(NL*PL*sizeof(double));
		double *COSTSaux=(double *)malloc(NL*PL*sizeof(double));
		if(XI==NULL || YI==NULL || ZI==NULL || COSTS==NULL || COSTSaux==NULL){
			fprintf(stderr,"\n Ran out of memory, exiting... \n");
			exit(1);
		}
		vnl_vector<double> aux;
		for(int l=0; l<PL; l++)
		{
			aux=contour+normals*(l*INC-SEARCH_DIST);
			for(int j=0; j<NL; j++)
			{
				XI[l*NL+j]=aux[j];
				YI[l*NL+j]=aux[j+NL];
				ZI[l*NL+j]=aux[j+2*NL];
			}
		}

		ImageType::Pointer Caux = ImageType::New();
		Caux->SetRegions( C->GetRequestedRegion() );
		Caux->CopyInformation( C );
		Caux->Allocate();
		IteratorType CauxIt( Caux, Caux->GetRequestedRegion() );
		CauxIt.GoToBegin();
		// IteratorType Iit(C,C->GetRequestedRegion() );
		IteratorType RegIt(registered,registered->GetRequestedRegion() );
		RegIt.GoToBegin();
		CIt.GoToBegin();
		PixelType val=0;
		while(!CIt.IsAtEnd())
		{
			val=RegIt.Get();
			if(val==OUT_CONSTANT)
				CauxIt.Set(10);
			else
				CauxIt.Set(CIt.Get());

			++CIt; ++CauxIt; ++RegIt;
		}


		LinearInterpolatorType::Pointer CauxInterp = LinearInterpolatorType::New();
		CauxInterp->SetInputImage(Caux);
		for(int i=0; i<NL*PL; i++)
		{
			if(XI[i]<1 || YI[i]<1 || ZI[i]<1 || XI[i]>imSize[1]  || YI[i]>imSize[0]  || ZI[i]>imSize[2]){
				COSTS[i]=0;
				COSTSaux[i]=10;
			}
			else
			{
				position[1]=XI[i]-1;	position[0]=YI[i]-1;	position[2]=ZI[i]-1;			
				COSTS[i]=Cinterp->EvaluateAtContinuousIndex(position);
				COSTSaux[i]=CauxInterp->EvaluateAtContinuousIndex(position);
			}
		}


		// kill profiles out of the main volume
		double new_prof[53]={0, -0.038462, -0.076923, -0.11538, -0.15385, -0.19231, -0.23077, -0.26923, -0.30769, -0.34615, -0.38462, -0.42308, -0.46154, -0.5, -0.53846, -0.57692, -0.61538, -0.65385, -0.69231, -0.73077, -0.76923, -0.80769, -0.84615, -0.88462, -0.92308, -0.96154, -1, -0.96154, -0.92308, -0.88462, -0.84615, -0.80769, -0.76923, -0.73077, -0.69231, -0.65385, -0.61538, -0.57692, -0.53846, -0.5, -0.46154, -0.42308, -0.38462, -0.34615, -0.30769, -0.26923, -0.23077, -0.19231, -0.15385, -0.11538, -0.076923, -0.038462, 0};
		for(int i=0; i<NL; i++)
		{
			if(COSTSaux[(PL-1)/2*NL+i]>0){
				for(int j=0; j<53; j++)
					COSTS[i+j*NL]=new_prof[j];				
			}
		}



		// prepare weights
		double *W=(double *)malloc(NL*PL*sizeof(double));
		if(W==NULL){
			fprintf(stderr,"\n Ran out of memory, exiting... \n");
			exit(1);
		}
		for(int ind=0; ind<PL-1; ind++)
		{
			for(int n=0; n<NL; n++)
			{
				W[ind*NL+n]=COSTS[ind*NL+n]-COSTS[(ind+1)*NL+n];
			}
		}
		for(int n=0; n<NL; n++)
		{
			W[(PL-1)*NL+n]=COSTS[(PL-1)*NL+n];
		}



		double *prA = NULL;
		int *irA = NULL;
		int *jcA = NULL;
		int nzels=load_EDGES(&prA,&irA,&jcA);


		double *prT = (double *) malloc(NL*PL*sizeof(double));
		int *irT = (int *) malloc(NL*PL*sizeof(int));
		if(prT==NULL || irT==NULL){
			fprintf(stderr,"\n Ran out of memory, exiting... \n");
			exit(1);
		}
		int count=0;
		double auxx;
		for (int j=0; j<NL*PL; j++){
			auxx=W[j];
			if(auxx<0)
			{
				prT[count]=-auxx;
				irT[count]=j;
				count++;
			}
		}
		int jcT[3]; jcT[0]=0; jcT[1]=count; jcT[2]=NL*PL;
		for (int j=0; j<NL*PL; j++){
			auxx=W[j];
			if(auxx>=0){
				if(auxx==0){
					prT[count]=1e-50;
					irT[count]=j;
				}
				else
				{
					prT[count]=auxx;
					irT[count]=j;
				}
				count++;
			}
		}


		int *labels=maxflow_main(prA,irA,jcA,prT,irT,jcT,NL*PL,nzels); // nzels should be 1193817


		double *shift=(double *) malloc(3*NL*sizeof(double));
		if(shift==NULL){
			fprintf(stderr,"\n Ran out of memory, exiting... \n");
			exit(1);
		}
		for(int n=0; n<NL; n++)
		{
			int i=n;
			int j=0;
			while(labels[i]!=0 && j<PL){
				i+=NL;
				j++;
			}
			if(j<PL)
				shift[n]=(j-(PL+1)/2-0.5)*INC;
			else
				shift[n]=(j-(PL+1)/2-0.5)*INC;

			shift[n+NL]=shift[n];
			shift[n+2*NL]=shift[n];
		}
		for(int n=0; n<3*NL; n++)
			shift[n]=shift[n]*normals[n];


		double *contour_ref=(double *) malloc(3*NL*sizeof(double));
		if(contour_ref==NULL){
			fprintf(stderr,"\n Ran out of memory, exiting... \n");
			exit(1);
		}
		for(int n=0; n<3*NL; n++)
			contour_ref[n]=contour[n]+shift[n];




		printf("Done! It took roughly %d seconds\n",(int)(time(NULL)-t_ini_step));
		std::cout << std::flush;





		/**************************/
		/* Step 8: mesh to volume */
		/**************************/

		std::cout << "Step 8 of 9: building volume from mesh..."  << std::flush;
		t_ini_step = time(NULL);

		double step=0.5;
		MaskImageType::Pointer V = MaskImageType::New();
		V->SetRegions( C->GetRequestedRegion() );
		V->CopyInformation( C );
		V->Allocate();


		MaskIteratorType Vit(V,V->GetRequestedRegion() );
		Vit.GoToBegin();
		while(!Vit.IsAtEnd()){
			Vit.Set(0);
			++Vit;
		}

		double v1_v[3]; vnl_vector<double> v1(v1_v,3);
		double v2_v[3]; vnl_vector<double> v2(v2_v,3);
		double v3_v[3]; vnl_vector<double> v3(v3_v,3);
		vnl_matrix<double> A1(3,3);
		vnl_matrix<double> A2(3,3);
		vnl_matrix<double> A(3,3);
		vnl_matrix<double> B(3,3);
		vnl_vector<double> aux2;
		vnl_vector<double> aux3;
		vnl_vector<double> v2p;
		vnl_vector<double> v3p;
		vnl_vector<double> d12;
		vnl_vector<double> d13;
		vnl_vector<double> d23;
		double thetaA1, thetaA2;
		double xg[200][200];
		double yg[200][200];
		bool map1[200][200];
		bool map2[200][200];
		bool map[200][200];
		double xpp[40000];
		double ypp[40000];
		vnl_matrix<double> pointsPP(3,1200);
		vnl_matrix<double> points;
		vnl_matrix<double> pC;
		vnl_matrix<double> pF;
		ImageType::IndexType pixelIndex;

		for(int f=0; f<N_FACES; f++)
		{

			int ff;
			ff=face.get(f,0); v1[0]=contour_ref[ff]; v1[1]=contour_ref[ff+NL]; v1[2]=contour_ref[ff+2*NL];
			ff=face.get(f,1); v2[0]=contour_ref[ff]; v2[1]=contour_ref[ff+NL]; v2[2]=contour_ref[ff+2*NL];
			ff=face.get(f,2); v3[0]=contour_ref[ff]; v3[1]=contour_ref[ff+NL]; v3[2]=contour_ref[ff+2*NL];

			v2p=v2-v1;
			v3p=v3-v1;
			thetaA1=atan2(v2p[1],v2p[2]);
			A1[0][0]=1; A1[0][1]=0; A1[0][2]=0; A1[1][0]=0; A1[1][1]=cos(thetaA1); A1[1][2]=-sin(thetaA1);	A1[2][0]=0; A1[2][1]=sin(thetaA1); A1[2][2]=cos(thetaA1);
			aux=A1*v2p;
			thetaA2=atan2(aux[2],aux[0]);
			A2[0][0]=cos(thetaA2); A2[0][1]=0; A2[0][2]=sin(thetaA2); A2[1][0]=0; A2[1][1]=1; A2[1][2]=0;	A2[2][0]=-sin(thetaA2); A2[2][1]=0; A2[2][2]=cos(thetaA2);
			A=A2*A1;
			vnl_vector<double> tmp=A*v2p;
			if(tmp[0]<0){
				A[0][0]=-A[0][0]; A[0][1]=-A[0][1]; A[0][2]=-A[0][2]; A[2][0]=-A[2][0]; A[2][1]=-A[2][1]; A[2][2]=-A[2][2];
			}
			aux2=A*v3p;
			double thetaB=-atan2(aux2[2],aux2[1]);
			B[0][0]=1; B[0][1]=0; B[0][2]=0; B[1][0]=0; B[1][1]=cos(thetaB); B[1][2]=-sin(thetaB);	B[2][0]=0; B[2][1]=sin(thetaB); B[2][2]=cos(thetaB);
			aux3=B*aux2;
			double y2=aux3[1];
			if(y2<0){
				B[1][0]=-B[1][0]; B[1][1]=-B[1][1]; B[1][2]=-B[1][2]; B[2][0]=-B[2][0]; B[2][1]=-B[2][1]; B[2][2]=-B[2][2];
				y2=-y2;
			}
			double x1=dot_product(A.get_row(0),v2p);
			double x2=aux3[0];
			double x_max=x1>x2?x1:x2;
			int sx=ceil(x_max/step);
			int sy=ceil(y2/step);
			int count=0;



			for(int s1=0; s1<sx; s1++){
				for(int s2=0; s2<sy; s2++){
					xg[s1][s2]=s2*step;
					yg[s1][s2]=s1*step;
					map1[s1][s2]=(yg[s1][s2]<=xg[s1][s2]*y2/x2);
					if(x2<x1)
					{
						map2[s1][s2]=(yg[s1][s2]<=y2-(y2/(x1-x2))*(xg[s1][s2]-x2));
					}
					else{
						if(x2==x1)
							map2[s1][s2]=(xg[s1][s2]<x1);
						else
							map2[s1][s2]=(yg[s1][s2]>(y2/(x2-x1))*(xg[s1][s2]-x1));
					}
					map[s1][s2]=(map1[s1][s2] && map2[s1][s2]);
					if(map[s1][s2]==true){
						xpp[count]=xg[s1][s2];
						ypp[count]=yg[s1][s2];
						count++;
					}
				}
			}



			for(int c=0; c<count; c++){
				pointsPP[0][c*3]=xpp[c]; pointsPP[0][c*3+1]=xpp[c]; pointsPP[0][c*3+2]=xpp[c];
				pointsPP[1][c*3]=ypp[c]; pointsPP[1][c*3+1]=ypp[c]; pointsPP[1][c*3+2]=ypp[c];
				pointsPP[2][c*3]=-0.5; pointsPP[2][c*3+1]=0; pointsPP[2][c*3+2]=0.5;
			}
			count=3*count;

			d13=v3-v1;
			d12=v2-v1;
			d23=v3-v2;

			points=(A.transpose())*(B.transpose())*pointsPP.extract(3,count);
			for(int c=0; c<count; c++)
				points.set_column(c,v1+points.get_column(c));


			aux=points.get_row(0);
			points.set_row(0,points.get_row(1)/1);
			points.set_row(1,aux/1);

			pC=points/1.0;
			pF=points/1.0;

			for(int r=0; r<3; r++){
				for(int c=0; c<count; c++){
					pC[r][c]=ceil(points[r][c])-1;
					pF[r][c]=floor(points[r][c])-1;
					if (pC[r][c]<0) pC[r][c]=0;
					if (pF[r][c]<0) pF[r][c]=0;
					if (pC[r][c]>imSize[r]-1) pC[r][c]=imSize[r]-1;
					if (pF[r][c]>imSize[r]-1) pF[r][c]=imSize[r]-1;
				}
			}






			for(int c=0; c<count; c++)
			{
				pixelIndex[0] = pC[0][c]; pixelIndex[1] = pC[1][c]; pixelIndex[2] = pC[2][c];
				V->SetPixel( pixelIndex, 1 );
				pixelIndex[0] = pC[0][c]; pixelIndex[1] = pC[1][c]; pixelIndex[2] = pF[2][c];
				V->SetPixel( pixelIndex, 1 );
				pixelIndex[0] = pC[0][c]; pixelIndex[1] = pF[1][c]; pixelIndex[2] = pC[2][c];
				V->SetPixel( pixelIndex, 1 );
				pixelIndex[0] = pC[0][c]; pixelIndex[1] = pF[1][c]; pixelIndex[2] = pF[2][c];
				V->SetPixel( pixelIndex, 1 );
				pixelIndex[0] = pF[0][c]; pixelIndex[1] = pC[1][c]; pixelIndex[2] = pC[2][c];
				V->SetPixel( pixelIndex, 1 );
				pixelIndex[0] = pF[0][c]; pixelIndex[1] = pC[1][c]; pixelIndex[2] = pF[2][c];
				V->SetPixel( pixelIndex, 1 );
				pixelIndex[0] = pF[0][c]; pixelIndex[1] = pF[1][c]; pixelIndex[2] = pC[2][c];
				V->SetPixel( pixelIndex, 1 );
				pixelIndex[0] = pF[0][c]; pixelIndex[1] = pF[1][c]; pixelIndex[2] = pF[2][c];
				V->SetPixel( pixelIndex, 1 );
			}

		}

		double cx,cy,cz,fx,fy,fz;
		for(int c=0; c<NL; c++)
		{
			cx=ceil(contour_ref[c+NL])-1; cy=ceil(contour_ref[c])-1; cz=ceil(contour_ref[c+2*NL])-1;
			fx=floor(contour_ref[c+NL])-1; fy=floor(contour_ref[c])-1; fz=floor(contour_ref[c+2*NL])-1;
			if(cx<0) cx=0;
			if(cy<0) cy=0;
			if(cz<0) cz=0;
			if(cx>imSize[0]-1) cx=imSize[0]-1;
			if(cy>imSize[1]-1) cy=imSize[1]-1;
			if(cz>imSize[2]-1) cz=imSize[2]-1;

			if(fx<0) fx=0;
			if(fy<0) fy=0;
			if(fz<0) fz=0;
			if(fx>imSize[0]-1) fx=imSize[0]-1;
			if(fy>imSize[1]-1) fy=imSize[1]-1;
			if(fz>imSize[2]-1) fz=imSize[2]-1;

			pixelIndex[0] = cx; pixelIndex[1] = cy; pixelIndex[2] = cz;
			V->SetPixel( pixelIndex, 1 );
			pixelIndex[0] = cx; pixelIndex[1] = cy; pixelIndex[2] = fz;
			V->SetPixel( pixelIndex, 1 );
			pixelIndex[0] = cx; pixelIndex[1] = fy; pixelIndex[2] = cz;
			V->SetPixel( pixelIndex, 1 );
			pixelIndex[0] = cx; pixelIndex[1] = fy; pixelIndex[2] = fz;
			V->SetPixel( pixelIndex, 1 );
			pixelIndex[0] = fx; pixelIndex[1] = cy; pixelIndex[2] = cz;
			V->SetPixel( pixelIndex, 1 );
			pixelIndex[0] = fx; pixelIndex[1] = cy; pixelIndex[2] = fz;
			V->SetPixel( pixelIndex, 1 );
			pixelIndex[0] = fx; pixelIndex[1] = fy; pixelIndex[2] = cz;
			V->SetPixel( pixelIndex, 1 );
			pixelIndex[0] = fx; pixelIndex[1] = fy; pixelIndex[2] = fz;
			V->SetPixel( pixelIndex, 1 );
		}


		int rad=1;
		bool readyM=false;
		ErodeFilterType::Pointer eroder;
		DilateFilterType::Pointer dilater;
		ConnectedFilterType::Pointer ccFilter;
		OrFilterType::Pointer orFilter;
		PadFilterType::Pointer padder;

		while(readyM==false)
		{



			eroder = ErodeFilterType::New();
			dilater = DilateFilterType::New();
			ccFilter=ConnectedFilterType::New();
			orFilter=OrFilterType::New();
			padder=PadFilterType::New();

			BallType kernel1, kernel2;
			BallType::SizeType ballSize1, ballSize2;
			eroder->SetForegroundValue(1);
			dilater->SetForegroundValue(1);
			eroder->SetBackgroundValue(0);
			dilater->SetBackgroundValue(0);
			ballSize1.Fill(rad); ballSize2.Fill(rad+2);
			kernel1.SetRadius(ballSize1); kernel2.SetRadius(ballSize2);
			kernel1.CreateStructuringElement(); kernel2.CreateStructuringElement();


			padder->SetInput(V);
			MaskImageType::SizeType lpad, upad; lpad.Fill(50); upad.Fill(50);
			padder->SetPadUpperBound(upad);
			padder->SetPadLowerBound(lpad);
			padder->SetConstant(0);
			padder->Update();

			dilater->SetInput(padder->GetOutput());
			dilater->SetKernel(kernel1);
			dilater->Update();

			MaskWriterType::Pointer maskWriter = MaskWriterType::New();

			ccFilter->SetInput(dilater->GetOutput());
			ccFilter->SetLower(0);
			ccFilter->SetUpper(0);
			ccFilter->SetReplaceValue(1);
			MaskImageType::IndexType seed; seed[0]=56; seed[1]=60; seed[2]=98;
			ccFilter->AddSeed(seed);
			ccFilter->Update();


			orFilter->SetInput1(ccFilter->GetOutput());
			orFilter->SetInput2(dilater->GetOutput());
			orFilter->Update();

			eroder->SetInput(orFilter->GetOutput());
			eroder->SetKernel(kernel2);
			eroder->Update();


			double vol=0;
			MaskIteratorType ErodedIt(eroder->GetOutput(),eroder->GetOutput()->GetRequestedRegion() );
			ErodedIt.GoToBegin();
			while(!ErodedIt.IsAtEnd()){
				if(ErodedIt.Get()>0)
					vol++;
				++ErodedIt;
			}

			if(vol>2000000.0)
				rad++;
			else
				readyM=true;
		}
		CropFilterType::Pointer	cropper=CropFilterType::New();
		MaskImageType::SizeType lpad, upad; lpad.Fill(50); upad.Fill(50);
		cropper->SetLowerBoundaryCropSize(lpad);
		cropper->SetUpperBoundaryCropSize(upad);
		cropper->SetInput(eroder->GetOutput());
		cropper->Update();


		MaskImageType::Pointer toWarp = cropper->GetOutput();


		MaskWriterType::Pointer maskWriter = MaskWriterType::New();


		printf("Done! It took roughly %d seconds\n",(int)(time(NULL)-t_ini_step));
		std::cout << std::flush;





		/*******************************/
		/*      Step 9: warp back      */
		/*******************************/

		std::cout << "Step 9 of 9: warping back to original space..."  << std::flush;
		t_ini_step = time(NULL);




  		MaskResampleFilterType::Pointer maskResample = MaskResampleFilterType::New();
		MaskNNInterpolatorType::Pointer mnnInterpolator = MaskNNInterpolatorType::New();

  		maskResample->SetTransform( finalTransformAffine->GetInverseTransform() );
  		maskResample->SetInput( toWarp );
    		maskResample->SetSize(    readerI->GetOutput()->GetLargestPossibleRegion().GetSize() );
  		maskResample->SetOutputOrigin(  readerI->GetOutput()->GetOrigin() );
  		maskResample->SetOutputSpacing( readerI->GetOutput()->GetSpacing() );
  		maskResample->SetOutputDirection( readerI->GetOutput()->GetDirection() );
  		maskResample->SetDefaultPixelValue( 0 );
  		maskResample->SetInterpolator(mnnInterpolator);
   		maskResample->Update();

		ReaderType::Pointer readerI2 = ReaderType::New();
		readerI2->SetFileName( inputFilename  );
		readerI2->Update(); 
		MaskImageFilterType::Pointer maskerFinal = MaskImageFilterType::New();
		maskerFinal->SetInput1(readerI2->GetOutput());
		maskerFinal->SetInput2(maskResample->GetOutput());
		maskerFinal->Update();
		WriterType::Pointer finalWriter=WriterType::New();
		finalWriter->SetInput(maskerFinal->GetOutput());
		finalWriter->SetFileName(outputFilename);
		finalWriter->Update();


		if(argc>3){
			MaskWriterType::Pointer mw =  MaskWriterType::New(); 
			mw->SetInput(maskResample->GetOutput());
			mw->SetFileName(argv[3]);
			mw->Update();
		}

		printf("Done! It took roughly %d seconds\n",(int)(time(NULL)-t_ini_step));
		std::cout << std::flush;



		printf("\n Output ready \n");
		printf("\n The whole process took roughly %d seconds\n",(int)(time(NULL)-t_ini_global));
		printf("\n Freeing memory and deleting temporary files...\n\n\n");
		std::cout << std::flush;

		free(lambda_v);
		free(lambda2_v);
		free(mean_normals_v);
		free(normals_v);
		free(mu_v);
		free(PHI_v);
		free(PHI2_v);
		free(XI);
		free(YI);
		free(ZI);
		free(COSTS);
		free(COSTSaux);
		free(W);
		free(prA); 
		free(irA); 
		free(jcA);
		free(labels);
		free(shift);
		free(contour_ref);



		// clear temporary dirs
		sprintf(command,"%s %s%s*.*",DEL_CMD,temp_dir,PATH_SEPARATOR);
		system(command);
		sprintf(command,"rmdir %s",temp_dir);
		system(command);
		sprintf(command,"rmdir %s",temp_dir_base);
		system(command);
		

		return EXIT_SUCCESS;

	}

	catch( itk::ExceptionObject & err ) 
	{ 
		cout << "ExceptionObject caught !" << endl; 
		cout << err << endl; 
		return EXIT_FAILURE;
	} 

}


