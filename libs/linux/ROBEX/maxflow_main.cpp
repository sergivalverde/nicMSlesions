#include <stdio.h>
#include "graph.h"

int * maxflow_main(double *prA, int *irA, int *jcA,
				  double *prT, int *irT, int *jcT, int size, int n_nonzero)
{


	// fetch its dimensions
	// actually, we must have m=n
	int m = size;
	int n = size;
	int nzmax = n_nonzero;

	double *pr = prA;
	int *ir = irA;
	int *jc = jcA;

	// create graph
	typedef Graph<float,float,float> GraphType;
	// estimations for number of nodes and edges - we know these exactly!
	GraphType *g = new GraphType(/*estimated # of nodes*/ n, /*estimated # of edges*/ jc[n]); 

	// add the nodes
	// NOTE: their indices are 0-based
	g->add_node(n);

	// traverse the adjacency matrix and add n-links
	unsigned int i, j, k;
	float v;
	for (j = 0; j < n; j++)
	{
		// this is a simple check whether there are non zero
		// entries in the j'th column
		if (jc[j] == jc[j+1])
		{
			continue; // nothing in this column
		}
		for (k = jc[j]; k <= jc[j+1]-1; k++)
		{
			i = ir[k];
			v = (float)pr[k];
			//mexPrintf("non zero entry: (%d,%d) = %.2f\n", i+1, j+1, v);
			g->add_edge(i, j, v, 0.0f);
		}
	}

	// traverse the terminal matrix and add t-links
	pr = prT;
	ir = irT;
	jc = jcT;
	for (j = 0; j <= 1; j++)
	{
		if (jc[j] == jc[j+1])
		{
			continue;
		}
		for (k = jc[j]; k <= jc[j+1]-1; k++)
		{
			i = ir[k];
			v = (float)pr[k];

			if (j == 0) // source weight
			{
				g->add_tweights(i, v, 0.0f);
			}
			else if (j == 1) // sink weight
			{
				g->add_tweights(i, 0.0f, v);
			}
		}
	}

	/*
	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
	double* flow = mxGetPr(plhs[0]);
	*flow = g->maxflow();
	*/
	
	
	g->maxflow();

	// figure out segmentation
	int *labels=(int *) malloc(n*sizeof(int));
	if(labels==NULL){
		fprintf(stderr,"\n Ran out of memory, exiting... \n");
		exit(1);
	}
	for (i = 0; i < n; i++)
	{
		labels[i] = g->what_segment(i);
	}

	// cleanup
	delete g;

	return labels;
}

