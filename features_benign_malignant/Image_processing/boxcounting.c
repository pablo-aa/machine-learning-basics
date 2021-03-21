///////////////////////////////////////////////////////////////
//   PERMISSAO PARA USAR LIVREMENTE DESDE QUE CITADA A FONTE
//   AUTOR: Prof. Dr. Rodrigo C. Guido
//   guido@ifsc.usp.br
//   IFSC/USP    2006
//////////////////////////////////////////////////////////////
//bibliotecas de funções de dimensão fractal

#include "boxcounting.h"

//p = vector of the rows or columns, n = size of the vector
double box_counting(double *p, long n){

	double *s = new double[n];
	for(long i=0;i<n;i++){
		s[i]=p[i];
	 
	}
	double menor=s[0];
	for(long i=1;i<n;i++)
		if(s[i]<menor)
			menor=s[i];

	for(long i=0;i<n;i++)
		s[i]-=menor;

	double maior=s[0];
	for(long i=1;i<n;i++)
		if(s[i]>maior)
			maior=s[i];

	for(long i=0;i<n;i++)
		s[i]=(s[i]/maior)*n;
	//signal fits now inside a square

	double*x = new double[(int)(log2(n))];
	double*y = new double[(int)(log2(n))];

	x[0]=n;
	y[0]=1;
	for(int i=1;i<(int)(log2(n));i++){
		x[i]=(int)(x[i-1]/2.0);
		y[i]=bc(s,n,(int)(x[i]));
    }

	for(int i=0;i<((int)(log2(n)));i++){
		x[i]=log2(x[i]);
		y[i]=log2(y[i]);
    }

	double sx=0;
	double sy=0;
	double sxy=0;
	double sx2=0;
	for(int i=0;i<((int)(log2(n)));i++){
		sx+=x[i];
		sy+=y[i];
		sxy+=x[i]*y[i];
		sx2+=x[i]*x[i];
    }
	double result;
	result = ((-((sx*sy-((int)(log2(n)))*sxy)/(sx*sx-((int)(log2(n)))*sx2))));   
	return result;
}

int bc(double *v,long t,int q){
	double maior;
	double menor;
	long c=0;
	for(int i=0;i<(int)((double)t/(double)q);i++){
		maior=v[0];
		menor=v[0];
		for(int j=i*q;j<(i+1)*q;j++){
			if(v[j]>maior)
				maior=v[j];
			if(v[j]<menor)
				menor=v[j];
		}
		c+=(int)(maior-menor+1);
	}
	return(c);
}

double log2(double x){
	return(log10(x)/log10(2));
}