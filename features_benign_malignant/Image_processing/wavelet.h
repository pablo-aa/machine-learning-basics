///////////////////////////////////////////////////////////////
//   PERMISSAO PARA USAR LIVREMENTE DESDE QUE CITADA A FONTE
//   AUTOR: Prof. Dr. Rodrigo C. Guido
//   guido@ifsc.usp.br
//   IFSC/USP    2006
//////////////////////////////////////////////////////////////
//wavelet.h
//bibliotecas de funcoes de transformada wavelet

#include<math.h>
//#include <opencv\cv.h>
//#include <opencv\cxcore.h>

//using namespace cv;
//using namespace std;

//-------------------------------------------------------------
void transformada_wavelet(double* f, long n, int nivel, char ordem, double h[], int ch)
{
double *g=new double[ch];
for(int i=0;i<ch;i++)
        {
        g[i]=h[ch-i-1];
        if(i%2!=0)
            g[i]*=-1;
        }
int cg=ch;
long j=0;
double* t = new double[n];
if(ordem=='n') // n de normal para wavelet
	{
	for(long i=0;i<n;i+=2)  //trend
	 	{
    	 	t[j]=0;
     	 	for(long k=0;k<ch;k++)
     	 		t[j]+=f[(i+k)%n]*h[k];
     	 	j++;
     	 	}
	for(long i=0;i<n;i+=2) //fluctuation
    	  	{
     	  	t[j]=0;
      	  	for(long k=0;k<cg;k++)
      	  		t[j]+=f[(i+k)%n]*g[k];
     	  	j++;
      	  	}
	}
else // i de invertido para wavelet packet
	{
	for(long i=0;i<n;i+=2) //fluctuation
    	  	{
     	  	t[j]=0;
      	  	for(long k=0;k<cg;k++)
      	  		t[j]+=f[(i+k)%n]*g[k];
     	  	j++;
      	  	}
	for(long i=0;i<n;i+=2)  //trend
	 	{
    	 	t[j]=0;
     	 	for(long k=0;k<ch;k++)
     	 		t[j]+=f[(i+k)%n]*h[k];
     	 	j++;
     	 	}
	}
for(long i=0;i<n;i++)
	f[i]=t[i];
nivel--;
n/=2;
delete(t);
if(nivel>0)	
        transformada_wavelet(&f[0],n,nivel,ordem,h,ch);
}
//-------------------------------------------------------------
void transformada_wavelet_packet(double* f, long n, int nivel, double h[], int ch)
{
long inicio=0;
long comprimento=n;
for(int i=1;i<=nivel;i++) // por exemplo, para nivel 5, vou chamar 5 vezes a fun?o de tranasformada, cada vez em n?el 1.
	{
	inicio=0;
	comprimento=(int)(n/pow(2,i-1));
	for(int j=0;j<pow(2,i-1);j++)
		{
		if(j%2==0)
			transformada_wavelet(&f[inicio],comprimento,1,'n',h,ch); // n de ordem normal: primeiro passa-baixa e depois passa-alta
		else
			transformada_wavelet(&f[inicio],comprimento,1,'i',h,ch); // i de invertido: primeiro passa-alta e depois passa-baixa
		inicio+=comprimento;
		}
	}
}
//--------------------------------------------------------
void transformada_wavelet_inversa(double* f, long n, int nivel, char ordem, double h[], int ch)
{
double* g=new double[ch];
for(int i=0;i<ch;i++)
        {
        g[i]=h[ch-i-1];
        if(i%2!=0)
            g[i]*=-1;
        }

double* sfi=new double[ch];
for(int i=0;i<ch;i+=2)
	sfi[i]=h[ch-2-i];
for(int i=0;i<ch;i+=2)
	sfi[i+1]=g[ch-2-i];
int csfi=ch;

double *wfi=new double[csfi];
for(int i=0;i<ch;i+=2)
	wfi[i]=h[ch-1-i];
for(int i=0;i<ch;i+=2)
	wfi[i+1]=g[ch-1-i];

long comprimento_do_subsinal=2*(long)((n)/(pow(2,nivel)));
double* subsinal=new double[comprimento_do_subsinal];
if(ordem=='n') //normal
	{
	for(long i=0;i<comprimento_do_subsinal;i+=2)
		{
        	subsinal[i]=f[(int)(i/2)];
		subsinal[i+1]=f[(int)(i/2)+(int)(comprimento_do_subsinal/2)];
		}
	}
else // i  // invertido  -> alguns casos da packet
	{
	for(long i=0;i<comprimento_do_subsinal;i+=2)
		{
	       	subsinal[i]=f[(int)(i/2)+(int)(comprimento_do_subsinal/2)];
			subsinal[i+1]=f[(int)(i/2)];
		}
	}

long start;
if(comprimento_do_subsinal>=csfi)
	{
	if(comprimento_do_subsinal-csfi > 0)
	    start=(comprimento_do_subsinal-csfi)+2;
	else
	    start=-(comprimento_do_subsinal-csfi)+2;
	}
else
	{
	long comprimento_matricial_do_sinal=2;
	while(comprimento_matricial_do_sinal<csfi)
		comprimento_matricial_do_sinal+=comprimento_matricial_do_sinal;
	start=comprimento_matricial_do_sinal-csfi+2;
	}

for(long j=0;j<comprimento_do_subsinal;j+=2)
        {
 	f[j]=0;
	f[j+1]=0;
     	for(int k=0;k<csfi;k++)
		{
		f[j]+=sfi[k]*subsinal[(start+k)%(comprimento_do_subsinal)];
		f[j+1]+=wfi[k]*subsinal[(start+k)%(comprimento_do_subsinal)];
		}
	start+=2;
	}
nivel--;
delete(subsinal);
if(nivel>0)
       transformada_wavelet_inversa(&f[0],n,nivel,ordem,h,ch);
}
//--------------------------------------------------------
void transformada_wavelet_packet_inversa(double* f, long n, int nivel, double h[], int ch)
{
long inicio=0;
long comprimento=n;
for(int i=nivel;i>=1;i--) // por exemplo, para nivel 5, vou chamar 5 vezes a fun?o de tranasformada, cada vez em n?el 1.
	{
	inicio=0;
	comprimento=(int)(n/pow(2,i-1));
	for(int j=0;j<pow(2,i-1);j++)
		{
		if(j%2==0)
			transformada_wavelet_inversa(&f[inicio],comprimento,1,'n',h,ch); // n de ordem normal: primeiro passa-baixa e depois passa-alta
		else
			transformada_wavelet_inversa(&f[inicio],comprimento,1,'i',h,ch); // i de invertido: primeiro passa-alta e depois passa-baixa
		inicio+=comprimento;
		}
	}
}
//-------------------------------------------------------------
void transformada_wavelet_bidimensional(double** f, long li, long ci, long n, long m, int nivel, char ordem_l, char ordem_c, double h[], int ch)
{
	
	double* vetor_linha=new double[m];
	for(int i=li;i<li+m;i++)
	{
		for(int j=ci;j<ci+m;j++)
			vetor_linha[j-ci]=f[i][j];
		transformada_wavelet(&vetor_linha[0],m,1,ordem_l,h,ch);
		for(int j=ci;j<ci+m;j++){
			f[i][j]=vetor_linha[j-ci];
			//printf("%f\n", f[i][j]);
		}
	}
	double* vetor_coluna=new double[n];
	for(int j=ci;j<ci+m;j++)
	{
		for(int i=li;i<li+n;i++)
			vetor_coluna[i-li]=f[i][j];
		transformada_wavelet(&vetor_coluna[0],n,1,ordem_c,h,ch);
		for(int i=li;i<li+n;i++){
			f[i][j]=vetor_coluna[i-li];
			
		}
	}
	nivel--;
	n/=2;
	m/=2;
	delete(vetor_linha);
	delete(vetor_coluna);
	if(nivel>0)
		transformada_wavelet_bidimensional(f, li, ci, n, m, nivel, ordem_l, ordem_c,h,ch);
}
//-------------------------------------------------------------
void transformada_wavelet_bidimensional_inversa(double** f, long li, long ci, long n, long m, int nivel, char ordem_l, char ordem_c, double h[], int ch)
{

long quantidade_de_linhas_da_submatriz=2*(long)((n)/(pow(2,nivel)));
long quantidade_de_colunas_da_submatriz=2*(long)((m)/(pow(2,nivel)));


double* vetor_linha=new double[quantidade_de_colunas_da_submatriz];
for(int i=li;i<li+quantidade_de_linhas_da_submatriz;i++)
	{
	for(int j=ci;j<ci+quantidade_de_colunas_da_submatriz;j++)
		vetor_linha[j-ci]=f[i][j];
	transformada_wavelet_inversa(&vetor_linha[0],quantidade_de_colunas_da_submatriz,1,ordem_l,h,ch);
	for(int j=ci;j<ci+quantidade_de_colunas_da_submatriz;j++)
		f[i][j]=vetor_linha[j-ci];
	}


double* vetor_coluna=new double[quantidade_de_linhas_da_submatriz];
for(int j=ci;j<ci+quantidade_de_colunas_da_submatriz;j++)
	{
	for(int i=li;i<li+quantidade_de_linhas_da_submatriz;i++)
		vetor_coluna[i-li]=f[i][j];
	transformada_wavelet_inversa(&vetor_coluna[0],quantidade_de_linhas_da_submatriz,1,ordem_c,h,ch);
	for(int i=li;i<li+quantidade_de_linhas_da_submatriz;i++)
		f[i][j]=vetor_coluna[i-li];
	}

nivel--;
delete(vetor_linha);
delete(vetor_coluna);
if(nivel>0)
	transformada_wavelet_bidimensional_inversa(f, li, ci, n, m, nivel, ordem_l, ordem_c,h,ch);
}
//---------------------------------------------------------------
void transformada_wavelet_packet_bidimensional(double** f, long n, long m, int nivel, double h[], int ch)
{
long inicio_da_linha;
long inicio_da_coluna;
long comprimento_da_linha;
long comprimento_da_coluna;

for(int k=1;k<=nivel;k++) // por exemplo, para nivel 5, vou chamar 5 vezes a fun?o de transformada, cada vez em n?el 1.
	{
	inicio_da_linha=0;
	inicio_da_coluna=0;

	comprimento_da_linha=(int)(m/pow(2,k-1));
	comprimento_da_coluna=(int)(n/pow(2,k-1));

	for(int i=0;i<pow(2,k-1);i++)
		{
		inicio_da_coluna=0;
		for(int j=0;j<pow(2,k-1);j++)
			{
			if((i%2==0)&&(j%2==0))
				transformada_wavelet_bidimensional(f,inicio_da_linha, inicio_da_coluna, comprimento_da_linha,comprimento_da_coluna,1,'n','n',h,ch);
			else if ((i%2==0)&&(j%2!=0))
				transformada_wavelet_bidimensional(f,inicio_da_linha, inicio_da_coluna, comprimento_da_linha,comprimento_da_coluna,1,'i','n',h,ch); 
			else if ((i%2!=0)&&(j%2==0))
				transformada_wavelet_bidimensional(f,inicio_da_linha, inicio_da_coluna, comprimento_da_linha,comprimento_da_coluna,1,'n','i',h,ch); 
			else
				transformada_wavelet_bidimensional(f,inicio_da_linha, inicio_da_coluna, comprimento_da_linha,comprimento_da_coluna,1,'i','i',h,ch);
			inicio_da_coluna+=comprimento_da_coluna;
			}
		inicio_da_linha+=comprimento_da_linha;
		}
	}
}
//-------------------------------------------------------------
void transformada_wavelet_packet_bidimensional_inversa(double** f, long n, long m, int nivel, double h[], int ch)
{
long inicio_da_linha;
long inicio_da_coluna;
long comprimento_da_linha;
long comprimento_da_coluna;

for(int k=nivel;k>=1;k--) 
	{
	inicio_da_linha=0;
	inicio_da_coluna=0;

	comprimento_da_linha=(int)(m/pow(2,k-1));
	comprimento_da_coluna=(int)(n/pow(2,k-1));

	for(int i=0;i<pow(2,k-1);i++)
		{
		inicio_da_coluna=0;
		for(int j=0;j<pow(2,k-1);j++)
			{
			if((i%2==0)&&(j%2==0))
				transformada_wavelet_bidimensional_inversa(f,inicio_da_linha, inicio_da_coluna, comprimento_da_linha,comprimento_da_coluna,1,'n','n',h,ch);
			else if ((i%2==0)&&(j%2!=0))
				transformada_wavelet_bidimensional_inversa(f,inicio_da_linha, inicio_da_coluna, comprimento_da_linha,comprimento_da_coluna,1,'i','n',h,ch); 
			else if ((i%2!=0)&&(j%2==0))
				transformada_wavelet_bidimensional_inversa(f,inicio_da_linha, inicio_da_coluna, comprimento_da_linha,comprimento_da_coluna,1,'n','i',h,ch); 
			else
				transformada_wavelet_bidimensional_inversa(f,inicio_da_linha, inicio_da_coluna, comprimento_da_linha,comprimento_da_coluna,1,'i','i',h,ch);
			inicio_da_coluna+=comprimento_da_coluna;
			}
		inicio_da_linha+=comprimento_da_linha;
		}
	}
}
//--------------------------------------------------------------------
