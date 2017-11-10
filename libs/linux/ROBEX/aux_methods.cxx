#include "aux_methods.h"
#include "swaps.h"
#include "stdio.h"
#include <cstdlib>
#include <iostream>
#include <vector>

#ifdef _WIN32
#define DAT_DIR ".\\dat\\"
#else
#define DAT_DIR "./dat/"
#endif


char * generateBogusFilename()
{
	char *filename =  (char *) malloc(20*sizeof(char));
	if(filename==NULL){
			fprintf(stderr,"\n Ran out of memory, exiting... \n");
			exit(1);
		}
	int i;
	int random_integer;
	char a='a';

	for(i=0; i<8; i++){
		random_integer = (rand()%26);
		filename[i]=a+random_integer;
	}

	filename[8]='\0';

	return filename;
}


int * randperm(int n_el){
	int *ordered=(int *) malloc(n_el*sizeof(int));
	int *random=(int *) malloc(n_el*sizeof(int));

	if(ordered==NULL || random==NULL){
			fprintf(stderr,"\n Ran out of memory, exiting... \n");
			exit(1);
		}

	for(int i=0; i<n_el; i++) {
		*(ordered+i)=i;
	}

	int ri;
	int remain=n_el;
	int i=0;
	while (remain>0) {
		ri = rand()%remain;
		random[i]=ordered[ri];
		i++;
		remain--;
		for(int j=ri; j<remain; j++)
			ordered[j]=ordered[j+1];
	}

  free(ordered);
  return random;
}


void cross_3d(double *res, vnl_vector<double> a, vnl_vector<double> b)
{
	*res=a[1]*b[2]-a[2]*b[1];
	*(res+1)=a[2]*b[0]-a[0]*b[2];
	*(res+2)=a[0]*b[1]-a[1]*b[0];
}


void fix_double(double *a){
	unsigned int *p=(unsigned int *)a;
	unsigned int *p2=p+1;
	unsigned int b=*p;
	unsigned int b2=*p2;
	b=( ((b) << 24) | \
	(((b) << 8) & 0x00ff0000) | \
	(((b) >> 8) & 0x0000ff00) | \
	((b) >> 24) );
	b2=( ((b2) << 24) | \
	(((b2) << 8) & 0x00ff0000) | \
	(((b2) >> 8) & 0x0000ff00) | \
	((b2) >> 24) );
	double res;
	unsigned int *pres=(unsigned int *)(&res);
	*pres=b2;
	*(pres+1)=b;
	*a=res;
}

void fix_int(int *a){
	unsigned int *p=(unsigned int *)a;
	unsigned int b=*p;
	b=( ((b) << 24) | \
	(((b) << 8) & 0x00ff0000) | \
	(((b) >> 8) & 0x0000ff00) | \
	((b) >> 24) );
	*a=(*((int *)(&b)));
}


int load_EDGES(double **pr, int **ir, int **jc){

	int nzel=1193817;

	*pr = (double *) malloc(nzel*sizeof(double));
	*ir = (int *) malloc(nzel*sizeof(int));
	*jc = (int *) malloc(171562*sizeof(jc));

	if(*pr==NULL || *ir==NULL || *jc==NULL){
		fprintf(stderr,"Insufficient memory available\n");
		exit(1);
	}

	FILE *fp;
	char filename[2000]; sprintf(filename,"%sEDGES_pr.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {
		fprintf(stderr,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}
	size_t result = fread (*pr,sizeof(double),nzel,fp);
	if (result != nzel) {fprintf(stderr,"Reading error\n"); exit (1);}
	fclose(fp);

	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	int twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<nzel; i++)
			fix_double(*pr+i);
	}




	sprintf(filename,"%sEDGES_ir.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {
		fprintf(stderr,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}
	result = fread (*ir,sizeof(int),nzel,fp);
	if (result != nzel) {fprintf(stderr,"Reading error\n"); exit (1);}
	fclose(fp);

	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<nzel; i++)
			fix_int(*ir+i);
	}



	sprintf(filename,"%sEDGES_jc.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {
		fprintf(stderr,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}
	result = fread (*jc,sizeof(int),171562,fp);
	if (result !=171562) {fprintf(stderr,"Reading error\n"); exit (1);}
	fclose(fp);

	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<171562; i++)
			fix_int(*jc+i);
	}

	return nzel;  // number of non-zero elements
}


int load_T(double **pr, int **ir, int **jc){

	int nzel=166576;

	*pr = (double *) malloc(nzel*sizeof(double));
	*ir = (int *) malloc(nzel*sizeof(int));
	*jc = (int *) malloc(3*sizeof(jc));

	if(*pr==NULL || *ir==NULL || *jc==NULL){
		fprintf(stderr,"Insufficient memory available\n");
		exit(1);
	}

	FILE *fp;
	char filename[2000]; sprintf(filename,"%sT_pr.dat",DAT_DIR);
	if( ( fp = fopen(filename, "rb")) == NULL) {
		fprintf(stderr,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}
	size_t result = fread (*pr,sizeof(double),nzel,fp);
	if (result != nzel) {fprintf(stderr,"Reading error\n"); exit (1);}
	fclose(fp);

	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	int twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<nzel; i++)
			fix_double(*pr+i);
	}

	sprintf(filename,"%sT_ir.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {
		fprintf(stderr,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}
	result = fread (*ir,sizeof(int),nzel,fp);
	if (result != nzel) {fprintf(stderr,"Reading error\n"); exit (1);}
	fclose(fp);

	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<nzel; i++)
			fix_int(*ir+i);
	}


	sprintf(filename,"%sT_jc.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {
		fprintf(stderr,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}
	result = fread (*jc,sizeof(int),3,fp);
	if (result !=3) {fprintf(stderr,"Reading error\n"); exit (1);}
	fclose(fp);

	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<3; i++)
			fix_int(*jc+i);
	}

	return nzel;  // number of non-zero elements
}


double *load_lambda(){
	double * p = (double *) malloc(19*sizeof(double));
	if(p==NULL){
		fprintf(stderr,"Insufficient memory available\n");
		exit(1);
	}
	FILE *fp;
	char filename[2000]; sprintf(filename,"%slambda.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {
		fprintf(stdout,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}

	size_t result = fread (p,sizeof(double),19,fp);
	if (result != 19) {fprintf(stdout,"Reading error\n"); exit (1);}
	fclose(fp);

	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	int twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<19; i++)
			fix_double(p+i);
	}

	return p;
}


double *load_lambda2(){
	double * p = (double *) malloc(19*sizeof(double));
	if(p==NULL){
		fprintf(stderr,"Insufficient memory available\n");
		exit(1);
	}
	FILE *fp;
		char filename[2000]; sprintf(filename,"%slambda2.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {
		fprintf(stdout,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}

	size_t result = fread (p,sizeof(double),19,fp);
	if (result != 19) {fprintf(stdout,"Reading error\n"); exit (1);}

	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	int twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<19; i++)
			fix_double(p+i);
	}

	fclose(fp);

	return p;
}


double *load_mean_normals(){
	double * p = (double *) malloc(9711*sizeof(double));
	if(p==NULL){
		fprintf(stderr,"Insufficient memory available\n");
		exit(1);
	}
	FILE *fp;
		char filename[2000]; sprintf(filename,"%smean_normals.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {
		fprintf(stdout,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}

	size_t result = fread (p,sizeof(double),9711,fp);
	if (result != 9711) {fprintf(stdout,"Reading error\n"); exit (1);}

	fclose(fp);


	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	int twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<9711; i++)
			fix_double(p+i);
	}

	return p;
}


double *load_mu(){
	double * p = (double *) malloc(9711*sizeof(double));
	if(p==NULL){
		fprintf(stderr,"Insufficient memory available\n");
		exit(1);
	}
	FILE *fp;
	char filename[2000]; sprintf(filename,"%smu.dat",DAT_DIR);
	if( ( fp = fopen(filename, "rb")) == NULL) {
		fprintf(stdout,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}

	size_t result = fread (p,sizeof(double),9711,fp);
	if (result != 9711) {fprintf(stdout,"Reading error\n"); exit (1);}

	fclose(fp);

	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	int twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<9711; i++)
			fix_double(p+i);
	}

	return p;
}


double *load_PHI(){
	double * p = (double *) malloc(184509*sizeof(double));
	if(p==NULL){
		fprintf(stderr,"Insufficient memory available\n");
		exit(1);
	}
	FILE *fp;
		char filename[2000]; sprintf(filename,"%sPHI.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {
		fprintf(stdout,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}

	size_t result = fread (p,sizeof(double),184509,fp);
	if (result != 184509) {fprintf(stdout,"Reading error\n"); exit (1);}

	fclose(fp);

	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	int twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<184509; i++)
			fix_double(p+i);
	}

	return p;
}


double *load_PHI2(){
	double * p = (double *) malloc(184509*sizeof(double));
	if(p==NULL){
		fprintf(stderr,"Insufficient memory available\n");
		exit(1);
	}
	FILE *fp;
	char filename[2000]; sprintf(filename,"%sPHI2.dat",DAT_DIR);
	if( ( fp = fopen(filename, "rb")) == NULL) {
		fprintf(stdout,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}

	size_t result = fread (p,sizeof(double),184509,fp);
	if (result != 184509) {fprintf(stdout,"Reading error\n"); exit (1);}

	fclose(fp);

	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	int twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<184509; i++)
			fix_double(p+i);
	}

	return p;
}


double *load_face(){
	double * p = (double *) malloc(6470*3*sizeof(double));
	if(p==NULL){
		fprintf(stderr,"Insufficient memory available\n");
		exit(1);
	}
	FILE *fp;
	char filename[2000]; sprintf(filename,"%sface.dat",DAT_DIR);
	if( ( fp = fopen(filename, "rb")) == NULL) {
		fprintf(stdout,"Cannot open a necessary .dat file\n");
		exit( 1 );
	}

	size_t result = fread (p,sizeof(double),6470*3,fp);
	if (result != 6470*3) {fprintf(stdout,"Reading error\n"); exit (1);}

	fclose(fp);


	sprintf(filename,"%s256.dat",DAT_DIR);
	if( ( fp = fopen( filename, "rb")) == NULL) {fprintf(stdout,"Cannot open a necessary .dat file\n"); exit( 1 );}
	int twoFiftySix=0;	result = fread (&twoFiftySix,sizeof(int),1,fp);	
	if (result != 1) {fprintf(stdout,"Reading error\n"); exit (1);}
	if(twoFiftySix!=256){
		for(int i=0; i<6470*3; i++)
			fix_double(p+i);
	}

	return p;
}



