int * maxflow_main(double *prA, int *irA, int *jcA, double *prT, int *irT, int *jcT, int size, int n_nonzero);
double estimate_sym_cost(itk::Image< float,3 >::Pointer Mirrored,
						 itk::ImageRegionIteratorWithIndex< itk::Image< float,3 > > MirroredIt,
						 itk::ImageRegionIteratorWithIndex< itk::Image< float,3 > > ImIt,
						 itk::LinearInterpolateImageFunction<itk::Image< float,3 >, double>::Pointer Iinterp,
						 vnl_vector<double> n, double m);
int load_T(double **pr, int **ir, int **jc);
