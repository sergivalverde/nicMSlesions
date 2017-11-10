Robust Brain Extraction (ROBEX)  v1.2
Author: Juan Eugenio Iglesias
Email: iglesias@nmr.mgh.harvard.edu
www: http://www.jeiglesias.com
--------------------------------------


Citation:

Iglesias JE, Liu CY, Thompson P, Tu Z: "Robust Brain Extraction Across
Datasets and Comparison with Publicly Available Methods", IEEE Transactions
on Medical Imaging, 30(9), 2011, 1617-1634.



Compilation instructions:

1. Download and install CMake from http://www.cmake.org

2. Download and compile ITK (www.itk.org). This version of ROBEX
was compiled with ITK-4.2, but any version on the ITK 4 family should
work (I have not tried, though). 

3. Use Cmake to open CMakeLists.txt and generate a makefile or IDE
project file. Use the generated file to compile the project. In Linux,
make sure that you set CMAKE_EXE_LINKER_FLAGS to -lstdc++. Aso, make
sure that the executable ROBEX(.exe) ends up in this folder.

4. Enjoy!

------------------------


Usage:    

WINDOWS:

  runROBEX.bat inputFile strippedFile [outputMaskFile] [seed]



LINUX:

  runROBEX.sh inputFile strippedFile [outputMaskFile] [seed] 


Important details:

- As opposed to previous versions, any volume that is correctly
oriented in World coordinates should be correctly stripped. If 
you get a completely wrong segmentation with the shape of a brain,
please let me know (iglesias@nmr.mgh.harvard.edu).

- Even though ROBEX will read/write any format supported by ITK,
use of Nifti format (.nii or .nii.gz), which can properly handle
the transforms that relate World and voxel coordinates, is *highly*
encouraged. If you have Analyze data, please convert it to Nifti
first and then make sure that the orientation is correct before
feeding the images to ROBEX. 

- Some changes have been introduced in the registration, so you
might not be able to accurately reproduce the results from the 
paper; to do so, please use an older version of ROBEX.

- The behavior of ROBEX is not deterministic. Therefore, you might
observe (very) slight differences between the outputs of different
runs on the same scan. If you need results that can be exactly
reproduced, you can provide ROBEX with a seed (which must be >=0).

- ROBEX is designed to be robust. If ROBEX doesn't produce decent 
results for a given volume, I'd very much appreciate it if you let
me know by email (iglesias@nmr.mgh.harvard.edu).

- Please read license.txt!


------------------------

If you want to play with the code: this is version 1.2 of the software
and there are plenty of constants that are hard coded; sorry about that!
We'll try to make newer versions easier to read and modify.




----------------
CHANGES WITH RESPECT TO VERSION 1.1
----------------- 

- ROBEX now handles orientation information properly.

- ROBEX does not rely on Elastix anymore.


----------------
CHANGES WITH RESPECT TO VERSION 1.0
----------------- 

- ROBEX does not depend on the random forest library anymore, and therefore
it is now completely BSD.

- ROBEX is now called from a wrapper such that it is not required to run 
the software from the ROBEX directory anymore. In Linux, this wrapper also
ensures that the Elastix directory is in the library path.

- The wrapper also adds the Elastix directory to the library path.

- Added the option of specifying a seed in the command line so that the 
output of the algorithm is deterministic (and therefore reproducible).













