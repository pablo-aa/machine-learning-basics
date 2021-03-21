/*
Algorithm: Wavelet property extraction
Author: Roberta Oliveira
Email: roberta.oliveira@fe.up.pt
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
//#include <time.h> 

//#include <opencv\cv.h>
//#include <opencv\highgui.h>
//#include <opencv\cxcore.h>
//#include "opencv2\core\core.hpp"
//#include "opencv2\features2d\features2d.hpp"
//#include "opencv2\highgui\highgui.hpp"
//#include "opencv2\nonfree\nonfree.hpp"
//#include "opencv2\nonfree\features2d.hpp"
//#include "opencv2\imgproc\imgproc.hpp" //border detector

//using namespace std;
//using namespace cv;

double calcwaveletfeature(double **channel, int row, int col);

double fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10;
double fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10;

//Caculate the wavelet features
double calcwaveletfeature(double **channel, int row, int col)
{
	//Calculate the energy
	double value1, value2, value3, value4, value5, value6, value7, value8, value9, value10;
	double sum10=0.0, sum9=0.0, sum8=0.0, sum7=0.0, sum6=0.0, sum5=0.0, sum4=0.0, sum3=0.0, sum2=0.0, sum1=0.0;
	double logar10=0.0, logar9=0.0, logar8=0.0, logar7=0.0, logar6=0.0, logar5=0.0, logar4=0.0, logar3=0.0, logar2=0.0, logar1=0.0;
	double energy1, energy2, energy3, energy4, energy5, energy6, energy7, energy8, energy9, energy10;
	double entropy1,  entropy2,  entropy3, entropy4, entropy5,  entropy6,  entropy7,  entropy8,  entropy9,  entropy10;
		
	for(int i=0;i<(int)(row/2.0);i++)
		for(int j=0;j<(int)(col/2.0);j++)
		{
			value10=pow(channel[i+(int)(row/2.0)][j],2);//III
			value9=pow(channel[i+(int)(row/2.0)][j+(int)(col/2.0)],2);//IV
			value8=pow(channel[i][j+(int)(col/2.0)],2);//II
			
			sum10+=value10;
			sum9+=value9;
			sum8+=value8;

			if (value10>0.00)
				logar10+=(value10*log(value10));
			if (value9>0.00)
				logar9+=(value9*log(value9));
			if (value8>0.00)
				logar8+=(value8*log(value8));
		}
	for(int i=0;i<(int)(row/4.0);i++)
		for(int j=0;j<(int)(col/4.0);j++)
		{
			value7=pow(channel[i+(int)(row/4.0)][j],2);//III
			value6=pow(channel[i+(int)(row/4.0)][j+(int)(col/4.0)],2);//IV
			value5=pow(channel[i][j+(int)(col/4.0)],2);//II

			sum7+=value7;
			sum6+=value6;
			sum5+=value5;   

			if (value7>0.00)
				logar7+=(value7*log(value7));
			if (value6>0.00)
				logar6+=(value6*log(value6));
			if (value5>0.00)
				logar5+=(value5*log(value5));  
		}        

	for(int i=0;i<(int)(row/8.0);i++)
		for(int j=0;j<(int)(col/8.0);j++)
		{
			value4=pow(channel[i+(int)(row/8.0)][j],2);//III
			value3=pow(channel[i+(int)(row/8.0)][j+(int)(col/8.0)],2);//IV
			value2=pow(channel[i][j+(int)(col/8.0)],2);//II
			value1=pow(channel[i][j],2);//I
			
			sum4+=value4;
			sum3+=value3;
			sum2+=value2;
			sum1+=value1;  	    

			if (value4>0.00)
				logar4+=(value4*log(value4));
			if (value3>0.00)
				logar3+=(value3*log(value3));
			if (value2>0.00)
				logar2+=(value2*log(value2));
			if (value1>0.00)
				logar1+=(value1*log(value1));
		}

	energy1=sqrt(sum1/((row/8.0)*(col/8.0)));
	energy2=sqrt(sum2/((row/8.0)*(col/8.0)));
	energy3=sqrt(sum3/((row/8.0)*(col/8.0)));
	energy4=sqrt(sum4/((row/8.0)*(col/8.0))); 
	energy5=sqrt(sum5/((row/4.0)*(col/4.0)));   
	energy6=sqrt(sum6/((row/4.0)*(col/4.0)));   
	energy7=sqrt(sum7/((row/4.0)*(col/4.0)));
	energy8=sqrt(sum8/((row/2.0)*(col/2.0)));      
	energy9=sqrt(sum9/((row/2.0)*(col/2.0)));      
	energy10=sqrt(sum10/((row/2.0)*(col/2.0))); 

	entropy1=logar1/((row/8.0)*(col/8.0));
	entropy2=logar2/((row/8.0)*(col/8.0));
	entropy3=logar3/((row/8.0)*(col/8.0));
	entropy4=logar4/((row/8.0)*(col/8.0)); 
	entropy5=logar5/((row/4.0)*(col/4.0));   
	entropy6=logar6/((row/4.0)*(col/4.0));   
	entropy7=logar7/((row/4.0)*(col/4.0));
	entropy8=logar8/((row/2.0)*(col/2.0));      
	entropy9=logar9/((row/2.0)*(col/2.0));      
	entropy10=logar10/((row/2.0)*(col/2.0));

	memcpy(&fenergy1, &energy1, sizeof(double));
	memcpy(&fenergy2, &energy2, sizeof(double));
	memcpy(&fenergy3, &energy3, sizeof(double));
	memcpy(&fenergy4, &energy4, sizeof(double));
	memcpy(&fenergy5, &energy5, sizeof(double));
	memcpy(&fenergy6, &energy6, sizeof(double));
	memcpy(&fenergy7, &energy7, sizeof(double));
	memcpy(&fenergy8, &energy8, sizeof(double));
	memcpy(&fenergy9, &energy9, sizeof(double));
	memcpy(&fenergy10, &energy10, sizeof(double));

	memcpy(&fentropy1, &entropy1, sizeof(double));
	memcpy(&fentropy2, &entropy2, sizeof(double));
	memcpy(&fentropy3, &entropy3, sizeof(double));
	memcpy(&fentropy4, &entropy4, sizeof(double));
	memcpy(&fentropy5, &entropy5, sizeof(double));
	memcpy(&fentropy6, &entropy6, sizeof(double));
	memcpy(&fentropy7, &entropy7, sizeof(double));
	memcpy(&fentropy8, &entropy8, sizeof(double));
	memcpy(&fentropy9, &entropy9, sizeof(double));
	memcpy(&fentropy10, &entropy10, sizeof(double));

	return 0;
}
