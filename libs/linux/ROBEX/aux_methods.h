#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>

char * generateBogusFilename();

int * randperm(int n_el);

void cross_3d(double *res, vnl_vector<double> a, vnl_vector<double> b);

int load_EDGES(double **pr, int **ir, int **jc);

double *load_lambda();

double *load_lambda2();

double *load_mean_normals();

double *load_mu();

double *load_PHI();

double *load_PHI2();

double *load_face();