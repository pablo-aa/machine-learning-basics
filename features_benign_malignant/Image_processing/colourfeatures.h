/*
Algorithm: Colour property extraction
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

double calcStatistic(int *channel, int countlesion);

double faverageC, fvarianceC, fdeviationC, fskewnessC;
int fminimumC, fmaximumC;

double calcStatistic(int *channel, int countlesion)
{
	int p;
	double sum, value1, value2;
	double average, variance, deviation, skewness;
	int minimum, maximum;

	sum=0, value1=0, value2=0;
    //Caculate the average
	for (p=1; p<=countlesion; p++)
		sum+=channel[p];
	average=sum/countlesion;

	//Caculate the sample variance
	for (p=1; p<=countlesion; p++)
		value1+=pow((channel[p]-average),2);
	variance=value1/(countlesion-1);

	//Calculate the standard deviation
	deviation = sqrt(variance);

	//Calculate the minimum value
	for (p=1; p<=countlesion; p++)
	{
		if (p==1)
			minimum=channel[p];
	
		if (channel[p]<minimum)
			minimum=channel[p];
	}

	//Calculate the maximum value
	for (p=1; p<=countlesion; p++)
	{
		if (p==1)
			maximum=channel[p];
		
		if (channel[p]>maximum)
			maximum=channel[p];
	}

	//Calculate the skewness
	for (p=1; p<=countlesion; p++)
		value2+=pow((channel[p]-average),3);
	skewness=(value2/countlesion)/pow(deviation,3);

	memcpy(&faverageC, &average, sizeof(double));
	memcpy(&fvarianceC, &variance, sizeof(double));
	memcpy(&fdeviationC, &deviation, sizeof(double));
	memcpy(&fminimumC, &minimum, sizeof(int));
	memcpy(&fmaximumC, &maximum, sizeof(int));
	memcpy(&fskewnessC, &skewness, sizeof(double));

	return 0;
}

