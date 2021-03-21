/*
Algorithm: feature extraction
Author: Roberta Oliveira
Email: roberta.oliveira@fe.up.pt
*/

#pragma warning(disable:4996)

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
// #include <malloc.h>
#include <math.h>
#include <string.h>
#include <time.h> 
#include <float.h>      // DBL_MAX
#include <limits>       // numeric_limits

// #include <opencv\cv.h>
// #include <opencv\highgui.h>
// #include <opencv\cxcore.h>
// #include "opencv\core\core.hpp"
// #include "opencv2\features2d\features2d.hpp"
// #include "opencv2\highgui\highgui.hpp"
// #include "opencv2\nonfree\nonfree.hpp"
// #include "opencv2\nonfree\features2d.hpp"
// #include "opencv2\imgproc\imgproc.hpp" //border detector

#include "shapefeatures.h"
#include "colourfeatures.h"
#include "boxcounting.h"
#include "CoOccurrence.h"
#include "wavelet.h"
#include "waveletfeatures.h"
//#include "csvparser.h"

using namespace std;
using namespace cv;

#define runs 1 //processing time parameters
#define total 1104 //Image parameters
#define malignant 188 //diagnosis


int main()
{
	//Local variables

	//Loop
	int num, img, i, j, iaux, p, s, d, featurenum, colmodel, t1, t2;
	//Read the images
	Mat gray, wgray, colourg; 
	Mat_<Vec3b> imageRGB, imageHSV, imageLAB, imageLUV,
		colourRGB, colourHSV, colourLAB, colourLUV,
		wimageRGB, wimageHSV, wimageLAB, wimageLUV;
	int row, col, wrow, wcol; 
	//Resize images
	//int dstWidth, dstHeight, auxcol, auxrow; 
	//Mat dst1, dst2, targetImage, targetRoi;
	//Shape
	int *x=NULL,*y=NULL,
		count, x1, x2, y1, y2, auxmajor;
	double major, distance;
	//Colour
	int *xi=NULL,*yj=NULL, *channel1c=NULL, *channel2c=NULL, *channel3c=NULL,
		countlesion;		
	//Fractal
	double *channel1f=NULL, *channel2f=NULL, *channel3f=NULL,
		rowresultC1, colresultC1, dimension1C1, dimension1rowC1, dimension1colC1, dimension2C1,
		rowresultC2, colresultC2, dimension1C2, dimension1rowC2, dimension1colC2, dimension2C2,
		rowresultC3, colresultC3, dimension1C3, dimension1rowC3, dimension1colC3, dimension2C3;	
	int countdim, countrow, countcol;
	//Wavelet		
	double **channel1w, **channel2w, **channel3w,
		h[]={0.7071067, 0.7071067}; //Haar filter
	int ch; 
	//Haralick
	int dist=1, quant=16; 
	//processing time 
	double mintime, maxtime, somatime, valortime, mediatime, deviationtime,
		seconds[10]={0};
	//Files
	char original[350], boundary[350],
		bmp[350], binary[350], contourn [350], outputFile[500];

	FILE *fptime, *fpfeatures, 
		*fpalgorithms_RGB, *fpalgorithms_HSV, *fpalgorithms_LAB, *fpalgorithms_LUV,
		*fpshape, *fpcolour, *fpfractal, *fpwavelet, *fpharalick,
		*fpcolour_RGB, *fpcolour_HSV, *fpcolour_LAB, *fpcolour_LUV,
		*fpfractal_RGB, *fpfractal_HSV, *fpfractal_LAB, *fpfractal_LUV,
		*fpwavelet_RGB, *fpwavelet_HSV, *fpwavelet_LAB, *fpwavelet_LUV,
		*fpharalick_RGB, *fpharalick_HSV, *fpharalick_LAB, *fpharalick_LUV;

	//r - reading
	//w - remove the data and save new data
	//a - add data after the previous data
	
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\timetexture.txt");
    fptime = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\features.txt");
    fpfeatures = fopen(outputFile, "a");

	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\shape.txt");
    fpshape = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\colour.txt");
    fpcolour = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\fractal.txt");
    fpfractal = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\wavelet.txt");
    fpwavelet = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\haralick.txt");
    fpharalick = fopen(outputFile, "a");

	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\algorithms\\algorithmsRGB.txt");
    fpalgorithms_RGB = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\algorithms\\algorithmsHSV.txt");
    fpalgorithms_HSV = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\algorithms\\algorithmsLAB.txt");
    fpalgorithms_LAB = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\algorithms\\algorithmsLUV.txt");
    fpalgorithms_LUV = fopen(outputFile, "a");

	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\colourRGB.txt");
    fpcolour_RGB = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\colourHSV.txt");
    fpcolour_HSV = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\colourLAB.txt");
    fpcolour_LAB = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\colourLUV.txt");
    fpcolour_LUV = fopen(outputFile, "a");

	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\fractalRGB.txt");
    fpfractal_RGB = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\fractalHSV.txt");
    fpfractal_HSV = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\fractalLAB.txt");
    fpfractal_LAB = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\fractalLUV.txt");
    fpfractal_LUV = fopen(outputFile, "a");

	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\waveletRGB.txt");
    fpwavelet_RGB = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\waveletHSV.txt");
    fpwavelet_HSV = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\waveletLAB.txt");
    fpwavelet_LAB = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\waveletLUV.txt");
    fpwavelet_LUV = fopen(outputFile, "a");

	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\haralickRGB.txt");
    fpharalick_RGB = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\haralickHSV.txt");
    fpharalick_HSV = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\haralickLAB.txt");
    fpharalick_LAB = fopen(outputFile, "a");
	sprintf(outputFile,"%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\Features\\models\\haralickLUV.txt");
    fpharalick_LUV = fopen(outputFile, "a");
    
	int classification[188]={3,12,21,25,28,29,30,33,34,38,41,44,47,52,54,71,74,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,
	161,162,163,164,232,233,234,235,236,237,238,239,240,241,242,243,244,310,311,317,318,319,323,329,334,335,349,351,354,365,367,370,393,396,397,398,399,400,408,410,424,425,426,427,428,429,430,431,436,
	437,438,439,440,441,442,443,444,445,446,447,448,449,450,610,622,631,636,639,645,651,672,676,679,680,689,703,704,707,714,718,724,725,730,731,740,748,754,755,760,773,775,776,794,802,811,812,818,826,
	827,830,831,846,851,852,855,861,866,867,874,892,894,900,930,937,939,944,956,961,963,964,975,978,988,992,995,997,998,999,1000,1001,1002,1012,1014,1017,1019,1024,1033,1058,1061,1079,1090};

	/*int classification[total];
	fpinput = fopen("C:\\Users\\pro12003\\Desktop\\Caracter�sticas\\classificacao.txt","r");
	for(i=0; i<total; i++)
	{
            fscanf(fpinput,"%d	%d", &j, &classification[i]);
	}
    fclose(fpinput);
	for(i=0; i<total; i++)
	{
            printf("%d\n", j);
			printf("%d\n", classification[i]);
	}*/

	//file, delimiter, first_line_is_header?
    /*CsvParser *csvparser = CsvParser_new("C:\\Users\\pro12003\\Desktop\\Caracter�sticas\\Feature_extraction\\Exemplos\\CsvParser\\examples\\example_file.csv", ",", 0);
    CsvRow *rowf;

    while ((rowf = CsvParser_getRow(csvparser)) ) {
    	printf("==NEW LINE==\n");
        const char **rowFields = CsvParser_getFields(rowf);
        for (i = 0 ; i < CsvParser_getNumFields(rowf) ; i++) {
            printf("FIELD: %s\n", rowFields[i]);
        }
		printf("\n");
        CsvParser_destroy_row(rowf);
    }
    CsvParser_destroy(csvparser);*/

	//Calculate the processing time
    for(num=1; num<=runs; num++)
	{
		clock_t start = clock();

		//for (img=601; img<=total; img++)
		for (img=1; img<=total; img++)
		{
			printf("Image %d\n", img);

			//sprintf (original, "%s%d%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\ISBI\\removed\\image\\image (",img,").jpg");
			//sprintf (boundary, "%s%d%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\ISBI\\removed\\boundary\\image (",img,").png");
			sprintf (original, "%s%d%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\ISBI\\square\\image\\",img,".bmp");
			sprintf (boundary, "%s%d%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\ISBI\\square\\roi\\",img,".bmp");

			//convert image extension for .bmp
			sprintf (bmp, "%s%d%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\ISBI\\bmp\\image\\",img,".bmp");
			sprintf (binary, "%s%d%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\ISBI\\bmp\\roi\\",img,".bmp");
			sprintf (contourn, "%s%d%s", "C:\\Users\\pro12003\\Desktop\\Feature_extraction\\Nome_Projeto\\ISBI\\bmp\\border\\",img,".bmp");

			Mat auximage = imread(bmp, 1);
			Mat auxroi = imread(binary, 0);
			Mat auxborder = imread(contourn, 0);

			//Wavelet method - the images must be square and each side must be power of 2
			Mat dst1 = imread(original, 1);
			Mat dst2 = imread(boundary, 0);

			//Wavelet method - the images must be square and each side must be power of 2
			Mat wimage=dst1.clone();
			Mat wroi=dst2.clone();
			wrow=wimage.rows, wcol=wimage.cols;

			dst1.release();
			dst2.release();
			//targetImage.release();
			//targetRoi.release();

			//Others methods
			Mat image=auximage.clone();
			Mat roi=auxroi.clone();
			Mat border=auxborder.clone();
			row=image.rows, col=image.cols;

			auximage.release();
			auxroi.release();
			auxborder.release();

			//Colour models
			cvtColor(image, gray, CV_RGB2GRAY);
			imageRGB=image.clone();
			cvtColor(image, imageHSV, CV_RGB2HSV);
			cvtColor(image, imageLAB, CV_RGB2Lab);
			cvtColor(image, imageLUV, CV_RGB2Luv);

			colourg=gray.clone();
			colourRGB=imageRGB.clone();
			colourHSV=imageHSV.clone();
			colourLAB=imageLAB.clone();
			colourLUV=imageLUV.clone();

			//Colour models - wavelet
			cvtColor(wimage, wgray, CV_RGB2GRAY);
			wimageRGB=wimage.clone();
			cvtColor(wimage, wimageHSV, CV_RGB2HSV);
			cvtColor(wimage, wimageLAB, CV_RGB2Lab);
			cvtColor(wimage, wimageLUV, CV_RGB2Luv);
	
			image.release();
			wimage.release();

			//Find the lesion's points
			for (i=0; i<row; i++)
				for (j=0; j<col; j++)
				{
					if (roi.at<uchar>(i,j) == 0)
					{
						gray.at<uchar>(i,j)=255; 

						imageRGB(i,j)[2]=255; 
						imageRGB(i,j)[1]=255;
						imageRGB(i,j)[0]=255;

						imageHSV(i,j)[2]=255; 
						imageHSV(i,j)[1]=255;
						imageHSV(i,j)[0]=255;

						imageLAB(i,j)[2]=255; 
						imageLAB(i,j)[1]=255;
						imageLAB(i,j)[0]=255;

						imageLUV(i,j)[2]=255; 
						imageLUV(i,j)[1]=255;
						imageLUV(i,j)[0]=255;
					}
				}
			for (i=0; i<wrow; i++)
				for (j=0; j<wcol; j++)
				{
					if (wroi.at<uchar>(i,j) == 0)
					{
						wgray.at<uchar>(i,j)=255; 

						wimageRGB(i,j)[2]=255; 
						wimageRGB(i,j)[1]=255;
						wimageRGB(i,j)[0]=255;

						wimageHSV(i,j)[2]=255; 
						wimageHSV(i,j)[1]=255;
						wimageHSV(i,j)[0]=255;

						wimageLAB(i,j)[2]=255; 
						wimageLAB(i,j)[1]=255;
						wimageLAB(i,j)[0]=255;

						wimageLUV(i,j)[2]=255; 
						wimageLUV(i,j)[1]=255;
						wimageLUV(i,j)[0]=255;
					}
				}
			//sprintf (boundary, "%s%d%s", "C:\\Users\\pro12003\\Desktop\\ISBI\\square\\roi\\0",img,".bmp");
			//imwrite (boundary, wimageRGB);
			wroi.release();

			//int vector (shape and colour)
			t1=row*col;
			t2=row+col;
			x=(int *)malloc(t1 * sizeof(int));
			y=(int *)malloc(t1 * sizeof(int));

			//int vector (colour)
			channel1c=(int *)malloc(t1 * sizeof(int));
			channel2c=(int *)malloc(t1 * sizeof(int));
			channel3c=(int *)malloc(t1 * sizeof(int));

			//double vector (fractal)
			channel1f=(double *)malloc(t2 * sizeof(double));
			channel2f=(double *)malloc(t2 * sizeof(double));
			channel3f=(double *)malloc(t2 * sizeof(double));

			for (featurenum=1; featurenum<=5; featurenum++) //5 feature extraction method (1-shape, 2-colour, 3-fractal, 4-wavelet, 5-haralick)
			{
				//char *feature;
				//Shape extraction

				if (featurenum==1)
				{
					//feature = "shape";
					
					//Find the coordinates of the border's points
					for (i=0; i<t1; i++)
					{
						x[i]=0;
						y[i]=0;
					}
					count=0;
					for (i=0; i<row; i++)
					{
						for (j=0; j<col; j++)
						{
							//Determine the countor perimeter
							if (border.at<uchar>(i,j)>0) 
							{
								count++;
								x[count]=i; 
								y[count]=j;
							}
						}
					}
					major=0;
					//Determine the greatest distance
					for (p=1; p<=count-1; p++)
					{
						for (s=p+1; s<=count; s++)
						{
							distance=sqrt(pow(x[p]-x[s],2)+pow(y[p]-y[s],2));
							if (distance>major) //It will be necessary to compare with the centroid
							{
								major=distance;
								//Points of the greatest distance
								x1=x[p];
								y1=y[p];
								x2=x[s];
								y2=y[s];
							}
						}
					}
					//Greatest diameter
					auxmajor=(int)(major);
					
					//Structural analysis/geometry-based features
					geometry(border, auxmajor);
					
					//Analyse spatial and central moments
					moments(border);
					
					//Asymmetry index - all perpendiculares based on the greatest distance
					asymmetryIndex(x, y, x1, y1, x2, y2, count, auxmajor);
					
					//Border irregularity
					borderIrregularity(x, y, x1, y1, x2, y2, count, row);
					
					if (num==1)
					{		
						fprintf (fpalgorithms_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,", lesion_area, perimeter, equi_diameter, compactness, circularity, solidity, rectangularity);
						fprintf (fpalgorithms_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,", lesion_area, perimeter, equi_diameter, compactness, circularity, solidity, rectangularity);
						fprintf (fpalgorithms_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,", lesion_area, perimeter, equi_diameter, compactness, circularity, solidity, rectangularity);
						fprintf (fpalgorithms_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,", lesion_area, perimeter, equi_diameter, compactness, circularity, solidity, rectangularity);
						fprintf (fpshape, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,", lesion_area, perimeter, equi_diameter, compactness, circularity, solidity, rectangularity);
						fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,", lesion_area, perimeter, equi_diameter, compactness, circularity, solidity, rectangularity);
						 
						fprintf (fpalgorithms_RGB, "%lf,%lf,", aspect_ratio, eccentricity);
						fprintf (fpalgorithms_HSV, "%lf,%lf,", aspect_ratio, eccentricity);
						fprintf (fpalgorithms_LAB, "%lf,%lf,", aspect_ratio, eccentricity);
						fprintf (fpalgorithms_LUV, "%lf,%lf,", aspect_ratio, eccentricity);
						fprintf (fpshape, "%lf,%lf,", aspect_ratio, eccentricity);
						fprintf (fpfeatures, "%lf,%lf,", aspect_ratio, eccentricity);
						
						fprintf (fpalgorithms_RGB, "%lf,%lf,%lf,", faverage, fvariance, fdeviation);
						fprintf (fpalgorithms_HSV, "%lf,%lf,%lf,", faverage, fvariance, fdeviation);
						fprintf (fpalgorithms_LAB, "%lf,%lf,%lf,", faverage, fvariance, fdeviation);
						fprintf (fpalgorithms_LUV, "%lf,%lf,%lf,", faverage, fvariance, fdeviation);
						fprintf (fpshape, "%lf,%lf,%lf,", faverage, fvariance, fdeviation);
						fprintf (fpfeatures, "%lf,%lf,%lf,", faverage, fvariance, fdeviation);

						fprintf (fpalgorithms_RGB, "%d,%d,%d,%d,%d,%d,", fvalley1, fpeak1, fline1, fvalley2, fpeak2, fline2);
						fprintf (fpalgorithms_HSV, "%d,%d,%d,%d,%d,%d,", fvalley1, fpeak1, fline1, fvalley2, fpeak2, fline2);
						fprintf (fpalgorithms_LAB, "%d,%d,%d,%d,%d,%d,", fvalley1, fpeak1, fline1, fvalley2, fpeak2, fline2);
						fprintf (fpalgorithms_LUV, "%d,%d,%d,%d,%d,%d,", fvalley1, fpeak1, fline1, fvalley2, fpeak2, fline2);
						fprintf (fpshape, "%d,%d,%d,%d,%d,%d,", fvalley1, fpeak1, fline1, fvalley2, fpeak2, fline2);
						fprintf (fpfeatures, "%d,%d,%d,%d,%d,%d,", fvalley1, fpeak1, fline1, fvalley2, fpeak2, fline2);

					}
					border.release();
					//printf ("shape end\n");
				}
				
				//Colour extration

				if (featurenum == 2)
				{	
					//feature = "colour";

					//Find the coordinates of the lesion's points
					for (i=0; i<t1; i++)
					{
						x[i]=0;
						y[i]=0;
					}
		
					countlesion = 0;
					for (i=0; i<row; i++)
					{
						for (j=0; j<col; j++)
						{
							if (roi.at<uchar>(i,j) > 0) {
								countlesion++;
								x[countlesion]=i; 
								y[countlesion]=j;
							}
						}
					}

					//Calculate statistics for the colour models
					for(colmodel=2; colmodel<=5; colmodel++) //5 colour models (1-Luminance, 2-RGB, 3-HSV, 4-LAB, 5-LUV)
					{
						//char *model;

						for (p=1; p<t1; p++)
						{
							channel1c[p]=0;
							channel2c[p]=0;
							channel3c[p]=0;
						}
						
						if (colmodel == 1)
						{	
							//model = "GRAY";
							for (p=1; p<=countlesion; p++)
								channel1c[p] = colourg.at<uchar>(xi[p],yj[p]);

							calcStatistic(channel1c, countlesion);
							if (num==1)
							{
								// Save all features in a file
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
						}
						
						if (colmodel == 2)
						{	
							//model = "RGB";
							
							for (p=1; p<=countlesion; p++)
							{
								channel1c[p] = colourRGB(x[p],y[p])[2];
								channel2c[p] = colourRGB(x[p],y[p])[1];
								channel3c[p] = colourRGB(x[p],y[p])[0];
							}
							//namedWindow("window", WINDOW_AUTOSIZE);
							//imshow("window", colour);
							calcStatistic(channel1c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_RGB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_RGB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
							calcStatistic(channel2c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_RGB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_RGB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
							calcStatistic(channel3c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_RGB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_RGB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
						}
						else if (colmodel == 3)
						{
							//model = "HSV";
							for (p=1; p<=countlesion; p++)
							{
								channel1c[p] = colourHSV(x[p],y[p])[2];
								channel2c[p] = colourHSV(x[p],y[p])[1];
								channel3c[p] = colourHSV(x[p],y[p])[0];
							}

							calcStatistic(channel1c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_HSV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_HSV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
							calcStatistic(channel2c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_HSV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_HSV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
							calcStatistic(channel3c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_HSV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_HSV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
						}
						else if (colmodel == 4)
						{
							//model = "LAB";
							for (int p=1; p<=countlesion; p++)
							{
								channel1c[p] = colourLAB(x[p],y[p])[2];
								channel2c[p] = colourLAB(x[p],y[p])[1];
								channel3c[p] = colourLAB(x[p],y[p])[0];
							}

							calcStatistic(channel1c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_LAB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_LAB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
							calcStatistic(channel2c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_LAB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_LAB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
							calcStatistic(channel3c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_LAB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_LAB, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
						}
						else if (colmodel == 5)
						{
							//model = "LUV";
							for (p=1; p<=countlesion; p++)
							{
								channel1c[p] = colourLUV(x[p],y[p])[2];
								channel2c[p] = colourLUV(x[p],y[p])[1];
								channel3c[p] = colourLUV(x[p],y[p])[0];
							} 

							calcStatistic(channel1c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_LUV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_LUV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
							calcStatistic(channel2c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_LUV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_LUV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
							calcStatistic(channel3c, countlesion);
							if (num==1)
							{
								fprintf (fpalgorithms_LUV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour_LUV, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpcolour, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
								fprintf (fpfeatures, "%lf,%lf,%lf,%d,%d,%lf,", faverageC, fvarianceC, fdeviationC, fminimumC, fmaximumC, fskewnessC);
							}
						}
					}
					colourg.release();
					colourRGB.release();
					colourHSV.release();
					colourLAB.release();
					colourLUV.release();
					//printf ("colour end\n");
				}

				//Fractal extration

				if (featurenum == 3)
				{	
					//feature = "fractal";
					
					//Calculate statistics for the colour models
					for(colmodel=2; colmodel<=5; colmodel++) //5 colour models
					{
						//char *model;

						for (p=0; p<t2; p++)
						{
							channel1f[p]=0;
							channel2f[p]=0;
							channel3f[p]=0;
						}

						count=0, countdim=0, countrow=0, countcol=0;
						rowresultC1=0, colresultC1=0, dimension1C1=0, dimension1rowC1=0, dimension1colC1=0, dimension2C1=0;
						rowresultC2=0, colresultC2=0, dimension1C2=0, dimension1rowC2=0, dimension1colC2=0, dimension2C2=0;
						rowresultC3=0, colresultC3=0, dimension1C3=0, dimension1rowC3=0, dimension1colC3=0, dimension2C3=0;
						
						count=0, countdim=0, dimension1C1=0, dimension1C2=0, dimension1C3=0;

						if (colmodel == 1)
						{
							//model = "GRAY";

							//Calculate the fractal dimension of lesion's rows
							for(i=0; i<row; i++)
							{
								for(j=0; j<col; j++) 
								{
									count++;
									channel1f[count]=gray.at<uchar>(i,j);
									
								}
								rowresultC1=box_counting(channel1f, count);
								
								//rows
								dimension1C1+=rowresultC1;
								countdim++;
								for(p=1; p<=count; p++){
									channel1f[p]=0;
								}
								count=0;
							}
							dimension1rowC1=dimension1C1;
							countrow=countdim;
							dimension1C1=0;
							countdim=0;
							
							//Calculate the fractal dimension of lesion's columns
							for(j=0; j<col; j++)
							{
								for(i=0; i<row; i++) 
								{
									count++;
									channel1f[count]=gray.at<uchar>(i,j);
								}
								colresultC1=box_counting(channel1f, count);
								//columns
								dimension1C1+=colresultC1;
								countdim++;
								for(p=1; p<=count; p++){
									channel1f[p]=0;
								}
								count=0;
							}
							dimension1colC1=dimension1C1;
							countcol=countdim;
							dimension1C1=0;
							countdim=0;
							dimension1C1=dimension1rowC1+dimension1colC1;
							countdim=countrow+countcol;

							// Calculate the dimension for image 2D (value between 2 and 3)
							dimension2C1=(dimension1C1/countdim)+1;
							if (num==1){
								fprintf(fpfractal, "%lf,", dimension2C1);
								fprintf(fpfeatures, "%lf,", dimension2C1);
							}
						}
						
						else if (colmodel == 2)
						{	
							//model = "RGB";
				
							//Calculate the fractal dimension of lesion's rows
							for(i=0; i<row; i++)
							{
								for(j=0; j<col; j++) 
								{
									count++;
									channel1f[count]=imageRGB(i,j)[2];
									channel2f[count]=imageRGB(i,j)[1];
									channel3f[count]=imageRGB(i,j)[0];
								}
								rowresultC1=box_counting(channel1f, count);
								rowresultC2=box_counting(channel2f, count);
								rowresultC3=box_counting(channel3f, count);
								
								//rows
								dimension1C1+=rowresultC1;
								dimension1C2+=rowresultC2;
								dimension1C3+=rowresultC3;
								countdim++;
								for(p=1; p<=count; p++){
									channel1f[p]=0;
									channel2f[p]=0;
									channel3f[p]=0;
								}
								count=0;
							}
							dimension1rowC1=dimension1C1;
							dimension1C1=0;
							dimension1rowC2=dimension1C2;
							dimension1C2=0;
							dimension1rowC3=dimension1C3;
							dimension1C3=0;
							countrow=countdim;
							countdim=0;

							//Calculate the fractal dimension of lesion's columns
							for(j=0; j<col; j++)
							{
								for(i=0; i<row; i++) 
								{
									count++;
									channel1f[count]=imageRGB(i,j)[2];
									channel2f[count]=imageRGB(i,j)[1];
									channel3f[count]=imageRGB(i,j)[0];
								}
								colresultC1=box_counting(channel1f, count);
								colresultC2=box_counting(channel2f, count);
								colresultC3=box_counting(channel3f, count);
								
								//columns
								dimension1C1+=colresultC1;
								dimension1C2+=colresultC2;
								dimension1C3+=colresultC3;
								countdim++;
								for(p=1; p<=count; p++){
									channel1f[p]=0;
									channel2f[p]=0;
									channel3f[p]=0;
								}
								count = 0;
							}
							dimension1colC1=dimension1C1;
							dimension1C1=0;
							dimension1colC2=dimension1C2;
							dimension1C2=0;
							dimension1colC3=dimension1C3;
							dimension1C3=0;
							countcol=countdim;
							countdim=0;
							dimension1C1=dimension1rowC1+dimension1colC1;
							dimension1C2=dimension1rowC2+dimension1colC2;
							dimension1C3=dimension1rowC3+dimension1colC3;
							countdim=countrow + countcol;
					
							// Calculate the dimension for image 2D (value between 2 and 3)
							dimension2C1=(dimension1C1/countdim)+1;
							dimension2C2=(dimension1C2/countdim)+1;
							dimension2C3=(dimension1C3/countdim)+1;
							if (num==1){
								fprintf(fpalgorithms_RGB, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfractal_RGB, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfractal, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfeatures, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
							}
						}

						else if (colmodel == 3)
						{	
							//model = "HSV";
				
							//Calculate the fractal dimension of lesion's rows
							for(i=0; i<row; i++)
							{
								for(j=0; j<col; j++) 
								{
									count++;
									channel1f[count]=imageHSV(i,j)[2];
									channel2f[count]=imageHSV(i,j)[1];
									channel3f[count]=imageHSV(i,j)[0];
								}
								rowresultC1=box_counting(channel1f, count);
								rowresultC2=box_counting(channel2f, count);
								rowresultC3=box_counting(channel3f, count);
								//rows
								dimension1C1+=rowresultC1;
								dimension1C2+=rowresultC2;
								dimension1C3+=rowresultC3;
								countdim++;
								for(p=1; p<=count; p++){
									channel1f[p]=0;
									channel2f[p]=0;
									channel3f[p]=0;
								}
								count=0;
							}
							dimension1rowC1=dimension1C1;
							dimension1C1=0;
							dimension1rowC2=dimension1C2;
							dimension1C2=0;
							dimension1rowC3=dimension1C3;
							dimension1C3=0;
							countrow=countdim;
							countdim=0;

							//Calculate the fractal dimension of lesion's columns
							for(j=0; j<col; j++)
							{
								for(i=0; i<row; i++) 
								{
									count++;
									channel1f[count]=imageHSV(i,j)[2];
									channel2f[count]=imageHSV(i,j)[1];
									channel3f[count]=imageHSV(i,j)[0];
								}
								colresultC1=box_counting(channel1f, count);
								colresultC2=box_counting(channel2f, count);
								colresultC3=box_counting(channel3f, count);
				
								//columns
								dimension1C1+=colresultC1;
								dimension1C2+=colresultC2;
								dimension1C3+=colresultC3;
								countdim++;
								for(p=1; p<=count; p++){
									channel1f[p]=0;
									channel2f[p]=0;
									channel3f[p]=0;
								}
								count = 0;
							}
							dimension1colC1=dimension1C1;
							dimension1C1=0;
							dimension1colC2=dimension1C2;
							dimension1C2=0;
							dimension1colC3=dimension1C3;
							dimension1C3=0;
							countcol=countdim;
							countdim=0;

							dimension1C1=dimension1rowC1+dimension1colC1;
							dimension1C2=dimension1rowC2+dimension1colC2;
							dimension1C3=dimension1rowC3+dimension1colC3;
							countdim=countrow + countcol;
					
							// Calculate the dimension for image 2D (value between 2 and 3)
							dimension2C1=(dimension1C1/countdim)+1;
							dimension2C2=(dimension1C2/countdim)+1;
							dimension2C3=(dimension1C3/countdim)+1;
							if (num==1){
								fprintf(fpalgorithms_HSV, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfractal_HSV, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfractal, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfeatures, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
							}
						}

						else if (colmodel == 4)
						{	
							//model = "LAB";

							//Calculate the fractal dimension of lesion's rows
							for(i=0; i<row; i++)
							{
								for(j=0; j<col; j++) 
								{
									count++;
									channel1f[count]=imageLAB(i,j)[2];
									channel2f[count]=imageLAB(i,j)[1];
									channel3f[count]=imageLAB(i,j)[0];
								}
								rowresultC1=box_counting(channel1f, count);
								rowresultC2=box_counting(channel2f, count);
								rowresultC3=box_counting(channel3f, count);
								//rows
								dimension1C1+=rowresultC1;
								dimension1C2+=rowresultC2;
								dimension1C3+=rowresultC3;
								countdim++;
								for(p=1; p<=count; p++){
									channel1f[p]=0;
									channel2f[p]=0;
									channel3f[p]=0;
								}
								count=0;
							}
							dimension1rowC1=dimension1C1;
							dimension1C1=0;
							dimension1rowC2=dimension1C2;
							dimension1C2=0;
							dimension1rowC3=dimension1C3;
							dimension1C3=0;
							countrow=countdim;
							countdim=0;

							//Calculate the fractal dimension of lesion's columns
							for(j=0; j<col; j++)
							{
								for(i=0; i<row; i++) 
								{
									count++;
									channel1f[count]=imageLAB(i,j)[2];
									channel2f[count]=imageLAB(i,j)[1];
									channel3f[count]=imageLAB(i,j)[0];
								}
								colresultC1=box_counting(channel1f, count);
								colresultC2=box_counting(channel2f, count);
								colresultC3=box_counting(channel3f, count);
				
								//columns
								dimension1C1+=colresultC1;
								dimension1C2+=colresultC2;
								dimension1C3+=colresultC3;
								countdim++;
								for(p=1; p<=count; p++){
									channel1f[p]=0;
									channel2f[p]=0;
									channel3f[p]=0;
								}
								count = 0;
							}
							dimension1colC1=dimension1C1;
							dimension1C1=0;
							dimension1colC2=dimension1C2;
							dimension1C2=0;
							dimension1colC3=dimension1C3;
							dimension1C3=0;
							countcol=countdim;
							countdim=0;

							dimension1C1=dimension1rowC1+dimension1colC1;
							dimension1C2=dimension1rowC2+dimension1colC2;
							dimension1C3=dimension1rowC3+dimension1colC3;
							countdim=countrow+countcol;
					
							// Calculate the dimension for image 2D (value between 2 and 3)
							dimension2C1=(dimension1C1/countdim)+1;
							dimension2C2=(dimension1C2/countdim)+1;
							dimension2C3=(dimension1C3/countdim)+1;
							if (num==1){
								fprintf(fpalgorithms_LAB, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfractal_LAB, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfractal, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfeatures, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
							}
						}

						else if (colmodel == 5)
						{	
							//model = "LUV";
				
							//Calculate the fractal dimension of lesion's rows
							for(i=0; i<row; i++)
							{
								for(j=0; j<col; j++) 
								{
									count++;
									channel1f[count]=imageLUV(i,j)[2];
									channel2f[count]=imageLUV(i,j)[1];
									channel3f[count]=imageLUV(i,j)[0];
								}
								rowresultC1=box_counting(channel1f, count);
								rowresultC2=box_counting(channel2f, count);
								rowresultC3=box_counting(channel3f, count);
								//rows
								dimension1C1+=rowresultC1;
								dimension1C2+=rowresultC2;
								dimension1C3+=rowresultC3;
								countdim++;
								for(p=1; p<=count; p++){
									channel1f[p]=0;
									channel2f[p]=0;
									channel3f[p]=0;
								}
								count=0;
							}
							dimension1rowC1=dimension1C1;
							dimension1C1=0;
							dimension1rowC2=dimension1C2;
							dimension1C2=0;
							dimension1rowC3=dimension1C3;
							dimension1C3=0;
							countrow=countdim;
							countdim=0;

							//Calculate the fractal dimension of lesion's columns
							for(j=0; j<col; j++)
							{
								for(i=0; i<row; i++) 
								{
									count++;
									channel1f[count]=imageLUV(i,j)[2];
									channel2f[count]=imageLUV(i,j)[1];
									channel3f[count]=imageLUV(i,j)[0];
								}
								colresultC1=box_counting(channel1f, count);
								colresultC2=box_counting(channel2f, count);
								colresultC3=box_counting(channel3f, count);
				
								//columns
								dimension1C1+=colresultC1;
								dimension1C2+=colresultC2;
								dimension1C3+=colresultC3;
								countdim++;
								for(p=1; p<=count; p++){
									channel1f[p]=0;
									channel2f[p]=0;
									channel3f[p]=0;
								}
								count = 0;
							}
							dimension1colC1=dimension1C1;
							dimension1C1=0;
							dimension1colC2=dimension1C2;
							dimension1C2=0;
							dimension1colC3=dimension1C3;
							dimension1C3=0;
							countcol=countdim;
							countdim=0;

							dimension1C1=dimension1rowC1+dimension1colC1;
							dimension1C2=dimension1rowC2+dimension1colC2;
							dimension1C3=dimension1rowC3+dimension1colC3;
							countdim=countrow + countcol;
					
							// Calculate the dimension for image 2D (value between 2 and 3)
							dimension2C1=(dimension1C1/countdim)+1;
							dimension2C2=(dimension1C2/countdim)+1;
							dimension2C3=(dimension1C3/countdim)+1;
							if (num==1){
								fprintf(fpalgorithms_LUV, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfractal_LUV, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfractal, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
								fprintf(fpfeatures, "%lf,%lf,%lf,", dimension2C1, dimension2C2, dimension2C3);
							}
						}
					}
					//printf ("fractal end\n");
				}
				
				//Wavelet extration

				if (featurenum == 4)
				{	
					//feature = "wavelet";
					
					//Filter used in the experiences. It must be always the filter h[] (low-pass ANALYSIS).
					//It can be put here the coefficients of any other wavelet filter.
					//double h[]={1.0/sqrt(2),1.0/sqrt(2)};
					//double h[]={0.7071067, 0.7071067}; //Haar filter
			
					//ch, which is defined automatically, is the size of h[]
					ch=sizeof(h)/sizeof(double); 
			
					//imagem, linha inicial, linha final, numero de linhas, numero de colunas, nivel de transformacao, 
					//ordem normal nas linhas, ordem normal nas colunas, filtro, tamanho do filtro
					//transformada_wavelet_bidimensional(auximg,0,0,row,col,3,'n','n',h,ch);
					
					//Convert images to double array to be able to apply the operations over them
					channel1w=new double*[wcol]; 
					channel2w=new double*[wcol]; 
					channel3w=new double*[wcol]; 
					for(p=0;p<wrow;p++)
					{
						channel1w[p]=new double[wrow];
						channel2w[p]=new double[wrow];
						channel3w[p]=new double[wrow];
					}

					//Calculate statistics for the colour models
					for(colmodel=2; colmodel<=5; colmodel++) //5 colour models
					{
						//char *model;

						for (i=0; i<wrow; i++)
							for (j=0; j<wcol; j++)
							{
							channel1w[i][j]=0;
							channel2w[i][j]=0;
							channel3w[i][j]=0;
							}
	
						if (colmodel == 1)
						{
							//model = "GRAY";

							//Return the value of image's pixel
							for (i=0; i<wrow; i++)
								for (j=0; j<wcol; j++)
									channel1w[i][j]=wgray.at<uchar>(i,j);

							transformada_wavelet_bidimensional(channel1w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel1w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel1w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);

							calcwaveletfeature(channel1w, wrow, wcol);
							if (num==1)
							{
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
						}

						else if (colmodel == 2)
						{	
							//model = "RGB";
	
							//Return the value of image's pixel
							for (i=0; i<wrow; i++)
								for (j=0; j<wcol; j++)
								{
									channel1w[i][j]=wimageRGB(i,j)[2];
									channel2w[i][j]=wimageRGB(i,j)[1];
									channel3w[i][j]=wimageRGB(i,j)[0];
								}
					
							transformada_wavelet_bidimensional(channel1w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel1w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel1w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);
					
							transformada_wavelet_bidimensional(channel2w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel2w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel2w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);

							transformada_wavelet_bidimensional(channel3w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel3w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel3w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);

							calcwaveletfeature(channel1w, wrow, wcol);	
							if (num==1)
							{
								fprintf (fpalgorithms_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
							calcwaveletfeature(channel2w, wrow, wcol);	
							if (num==1)
							{
								fprintf (fpalgorithms_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
							calcwaveletfeature(channel3w, wrow, wcol);	
							if (num==1)
							{
								fprintf (fpalgorithms_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_RGB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
						}

						else if (colmodel == 3)
						{
							//model = "HSV";

							//Return the value of image's pixel
							for (i=0; i<wrow; i++)
								for (j=0; j<wcol; j++)
								{
									channel1w[i][j]=wimageHSV(i,j)[2];
									channel2w[i][j]=wimageHSV(i,j)[1];
									channel3w[i][j]=wimageHSV(i,j)[0];
								}
					
							transformada_wavelet_bidimensional(channel1w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel1w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel1w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);
					
							transformada_wavelet_bidimensional(channel2w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel2w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel2w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);

							transformada_wavelet_bidimensional(channel3w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel3w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel3w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);

							calcwaveletfeature(channel1w, wrow, wcol);		
							if (num==1)
							{
								fprintf (fpalgorithms_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
							calcwaveletfeature(channel2w, wrow, wcol);	
							if (num==1)
							{
								fprintf (fpalgorithms_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
							calcwaveletfeature(channel3w, wrow, wcol);	
							if (num==1)
							{
								fprintf (fpalgorithms_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_HSV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
						}

						else if (colmodel == 4)
						{
							//model = "LAB";
	
							//Return the value of image's pixel
							for (int i=0; i<wrow; i++)
								for (int j=0; j<wcol; j++)
								{
									channel1w[i][j]=wimageLAB(i,j)[2];
									channel2w[i][j]=wimageLAB(i,j)[1];
									channel3w[i][j]=wimageLAB(i,j)[0];
								}
					
							transformada_wavelet_bidimensional(channel1w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel1w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel1w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);
					
							transformada_wavelet_bidimensional(channel2w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel2w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel2w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);

							transformada_wavelet_bidimensional(channel3w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel3w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel3w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);

							calcwaveletfeature(channel1w, wrow, wcol);	
							if (num==1)
							{
								fprintf (fpalgorithms_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
							calcwaveletfeature(channel2w, wrow, wcol);	
							if (num==1)
							{
								fprintf (fpalgorithms_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
							calcwaveletfeature(channel3w, wrow, wcol);	
							if (num==1)
							{
								fprintf (fpalgorithms_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_LAB, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
						}

						else if (colmodel == 5)
						{
							//model = "LUV";
	
							//Return the value of image's pixel
							for (i=0; i<wrow; i++)
								for (j=0; j<wcol; j++)
								{
									channel1w[i][j]=wimageLUV(i,j)[2];
									channel2w[i][j]=wimageLUV(i,j)[1];
									channel3w[i][j]=wimageLUV(i,j)[0];
								}
					
							transformada_wavelet_bidimensional(channel1w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel1w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel1w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);
					
							transformada_wavelet_bidimensional(channel2w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel2w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel2w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);

							transformada_wavelet_bidimensional(channel3w,0,0,wrow,wcol,1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel3w,0,0,(int)(wrow/2.0),(int)(wcol/2.0),1,'n','n',h,ch);
							transformada_wavelet_bidimensional(channel3w,0,0,(int)(wrow/4.0),(int)(wcol/4.0),1,'n','n',h,ch);

							calcwaveletfeature(channel1w, wrow, wcol);			
							if (num==1)
							{
								fprintf (fpalgorithms_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
							calcwaveletfeature(channel2w, wrow, wcol);	
							if (num==1)
							{
								fprintf (fpalgorithms_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
							calcwaveletfeature(channel3w, wrow, wcol);	
							if (num==1)
							{
								fprintf (fpalgorithms_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpalgorithms_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet_LUV, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpwavelet, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fenergy1, fenergy2, fenergy3, fenergy4, fenergy5, fenergy6, fenergy7, fenergy8, fenergy9, fenergy10);
								fprintf (fpfeatures, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", fentropy1, fentropy2, fentropy3, fentropy4, fentropy5, fentropy6, fentropy7, fentropy8, fentropy9, fentropy10);
							}
						}
					}
					wgray.release();
					wimageRGB.release();
					wimageHSV.release();
					wimageLAB.release();
					wimageLUV.release();
					//printf ("wavelet end\n");
				}

				//Haralick extration
	
				if (featurenum == 5)
				{	
					//feature = "haralick";

					Mat channel1h(row, col, CV_8UC1); 
					Mat channel2h(row, col, CV_8UC1); 
					Mat channel3h(row, col, CV_8UC1);

					CoOccurrenceExtractor c(row, col, dist, quant);

					for(colmodel=2; colmodel<=5; colmodel++) //5 colour models
					{
						//char *model; 

						for (i=0; i<row; i++)
							for (j=0; j<col; j++)
							{
								channel1h.at<uchar>(i,j)=0;
								channel2h.at<uchar>(i,j)=0;
								channel3h.at<uchar>(i,j)=0;
							}

						if (colmodel == 1)
						{
							//model = "GRAY";

							for (i=0; i<row; i++)
								for (j=0; j<col; j++)
									channel1h.at<uchar>(i,j)=gray.at<uchar>(i,j);
								
							c.setImage(channel1h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
								
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
						}
						else if (colmodel == 2)
						{	
							//model = "RGB";
							//split (colour, channel);

							for (i=0; i<row; i++)
								for (j=0; j<col; j++)
								{
									channel1h.at<uchar>(i,j)=imageRGB(i,j)[2];
									channel2h.at<uchar>(i,j)=imageRGB(i,j)[1];
									channel3h.at<uchar>(i,j)=imageRGB(i,j)[0];
								}
							
							c.setImage(channel1h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_RGB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_RGB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
							c.setImage(channel2h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_RGB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_RGB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
							c.setImage(channel3h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_RGB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_RGB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
						}
						else if (colmodel == 3)
						{	
							//model = "HSV";

							for (i=0; i<row; i++)
								for (j=0; j<col; j++)
								{
									channel1h.at<uchar>(i,j)=imageHSV(i,j)[2];
									channel2h.at<uchar>(i,j)=imageHSV(i,j)[1];
									channel3h.at<uchar>(i,j)=imageHSV(i,j)[0];
								}
						
							c.setImage(channel1h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_HSV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_HSV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
							c.setImage(channel2h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_HSV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_HSV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
							c.setImage(channel3h, roi);
							for(unsigned d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_HSV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_HSV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
						}
						else if (colmodel == 4)
						{	
							//model = "LAB";

							for (i=0; i<row; i++)
								for (j=0; j<col; j++)
								{
									channel1h.at<uchar>(i,j)=imageLAB(i,j)[2];
									channel2h.at<uchar>(i,j)=imageLAB(i,j)[1];
									channel3h.at<uchar>(i,j)=imageLAB(i,j)[0];
								}
						
							c.setImage(channel1h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_LAB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_LAB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
							c.setImage(channel2h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_LAB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_LAB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
							c.setImage(channel3h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_LAB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_LAB, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
						}
						else if (colmodel == 5)
						{	
							//model = "LUV";

							for (i=0; i<row; i++)
								for (j=0; j<col; j++)
								{
									channel1h.at<uchar>(i,j)=imageLUV(i,j)[2];
									channel2h.at<uchar>(i,j)=imageLUV(i,j)[1];
									channel3h.at<uchar>(i,j)=imageLUV(i,j)[0];
								}
						
							c.setImage(channel1h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_LUV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_LUV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
							c.setImage(channel2h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_LUV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_LUV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
							c.setImage(channel3h, roi);
							for(d=0; d<=4; d++)//the 4 directions + the normalized matrix
							{
								double Energy = c.getEnergy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Contrast = c.getContrast(CoOccurrenceExtractor::MatrixOrientation(d));
								double Correlation = c.getCorrelation(CoOccurrenceExtractor::MatrixOrientation(d));
								double Variance = c.getVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double InverseDifferenceMoment = c.getInverseDifferenceMoment(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumAverage = c.getSumAverage(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumVariance = c.getSumVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double SumEntropy = c.getSumEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double Entropy = c.getEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceVariance = c.getDifferenceVariance(CoOccurrenceExtractor::MatrixOrientation(d));
								double DifferenceEntropy = c.getDifferenceEntropy(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation1 = c.getMeasureOfCorrelation1(CoOccurrenceExtractor::MatrixOrientation(d));
								double MeasureOfCorrelation2 = c.getMeasureOfCorrelation2(CoOccurrenceExtractor::MatrixOrientation(d));
								double MaximalCorrelationCoefficient = c.getMaximalCorrelationCoefficient(CoOccurrenceExtractor::MatrixOrientation(d));
						
								//Save only the normalized matrix 
								if (d==4 && num==1)
								{
									fprintf (fpalgorithms_LUV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick_LUV, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpharalick, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
									fprintf (fpfeatures, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,", Energy, Contrast, Correlation, Variance, InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy, Entropy, DifferenceVariance, DifferenceEntropy, MeasureOfCorrelation1, MeasureOfCorrelation2, MaximalCorrelationCoefficient);
								}
							}
						} //End 5 colour model
					} //End colour models
					channel1h.release();
					channel2h.release();
					channel3h.release();
					//printf ("haralick end\n");
				} //End Haralick (feature 5)
			} //End features 
			gray.release();
			imageRGB.release();
			imageHSV.release();
			imageLAB.release();
			imageLUV.release();

			roi.release();
			
			//Close the vectors
			if (x!=NULL)
			{
				free (x);
				x=NULL;
			}
			if (y!=NULL)
			{
				free (y);
				y=NULL;
			}

			if (channel1c!=NULL)
			{
				free (channel1c);
				channel1c=NULL;
			}
			if (channel2c!=NULL)
			{
				free (channel2c);
				channel2c=NULL;
			}
			if (channel3c!=NULL)
			{
				free (channel3c);
				channel3c=NULL;
			}
			
			if (channel1f!=NULL)
			{
				free (channel1f);
				channel1f=0;
			}
			if (channel2f!=NULL)
			{
				free (channel2f);
				channel2f=0;
			}
			if (channel3f!=NULL)
			{
				free (channel3f);
				channel3f=0;
			}

			//Insert the classes
			
			//Space between the images
			if (num==1)
			{
				int diagnosis=0;
				for (p=0; p<malignant; p++)
				{
					if (classification[p]==img)
						diagnosis=1;
				}
				//fprintf (fpfeatures, "%d \n", classification[img-1]);
				//fprintf (fpfeatures, "\n");

				fprintf (fpfeatures, "%d\n", diagnosis);
				fprintf (fpshape, "%d\n", diagnosis);
				fprintf (fpcolour, "%d\n", diagnosis);
				fprintf (fpfractal, "%d\n", diagnosis);
				fprintf (fpwavelet, "%d\n", diagnosis);
				fprintf (fpharalick, "%d\n", diagnosis);

				fprintf (fpalgorithms_RGB, "%d\n", diagnosis);
				fprintf (fpalgorithms_HSV, "%d\n", diagnosis);
				fprintf (fpalgorithms_LAB, "%d\n", diagnosis);
				fprintf (fpalgorithms_LUV, "%d\n", diagnosis);

				fprintf (fpcolour_RGB, "%d\n", diagnosis);
				fprintf (fpcolour_HSV, "%d\n", diagnosis);
				fprintf (fpcolour_LAB, "%d\n", diagnosis);
				fprintf (fpcolour_LUV, "%d\n", diagnosis);

				fprintf (fpfractal_RGB, "%d\n", diagnosis);
				fprintf (fpfractal_HSV, "%d\n", diagnosis);
				fprintf (fpfractal_LAB, "%d\n", diagnosis);
				fprintf (fpfractal_LUV, "%d\n", diagnosis);

				fprintf (fpwavelet_RGB, "%d\n", diagnosis);
				fprintf (fpwavelet_HSV, "%d\n", diagnosis);
				fprintf (fpwavelet_LAB, "%d\n", diagnosis);
				fprintf (fpwavelet_LUV, "%d\n", diagnosis);

				fprintf (fpharalick_RGB, "%d\n", diagnosis);
				fprintf (fpharalick_HSV, "%d\n", diagnosis);
				fprintf (fpharalick_LAB, "%d\n", diagnosis);
				fprintf (fpharalick_LUV, "%d\n", diagnosis);
			}
		} //End images
		//Calculate the processing time 
		clock_t end = clock();
		seconds [num] = (double)(end - start) / CLOCKS_PER_SEC;
		//printf("TIME (seconds) %.5f\n", seconds[num]);
		
	}
	//Calculate the time average, minimum time, maximum time and time standard deviation
	mintime=999999, maxtime=-1;
	somatime=0, valortime=0;
	for(iaux=1; iaux<=runs; iaux++)
	{
	    somatime+=seconds[iaux];

	    if (seconds[iaux]<mintime)
	    	mintime = seconds[iaux];	
		
		if (seconds[iaux]>maxtime)
	    	maxtime = seconds[iaux];	
	}
	
	mediatime=somatime/runs;
	      
	for(iaux=1; iaux<=runs; iaux++)
		valortime += pow (seconds [iaux] - mediatime, 2);
	
	if (runs>1)
		deviationtime = sqrt(valortime/(runs - 1));
	else
		deviationtime = 0;
		      
    //Save data about the computation times
	fprintf(fptime, "Feature extraction - Computation times\n\n");
	fprintf(fptime, "Runs: %d\n\n", runs);
	fprintf(fptime, "Images: %d\n\n", total);
    fprintf(fptime, "Average time (seconds): %lf\n\n", mediatime);
    fprintf(fptime, "Standard deviation (seconds): %lf\n\n", deviationtime);
    fprintf(fptime, "Minimum time (seconds): %lf\n\n", mintime); 
	fprintf(fptime, "Maximum time (seconds): %lf\n\n", maxtime);

	fclose (fptime);
	fclose (fpfeatures);
	fclose (fpshape);
	fclose (fpcolour);
	fclose (fpfractal);
	fclose (fpwavelet);
	fclose (fpharalick);

	fclose (fpalgorithms_RGB);
	fclose (fpalgorithms_HSV);
	fclose (fpalgorithms_LAB);
	fclose (fpalgorithms_LUV);

	fclose (fpcolour_RGB);
	fclose (fpcolour_HSV);
	fclose (fpcolour_LAB);
	fclose (fpcolour_LUV);
	fclose (fpfractal_RGB);
	fclose (fpfractal_HSV);
	fclose (fpfractal_LAB);
	fclose (fpfractal_LUV);
	fclose (fpwavelet_RGB);
	fclose (fpwavelet_HSV);
	fclose (fpwavelet_LAB);
	fclose (fpwavelet_LUV);
	fclose (fpharalick_RGB);
	fclose (fpharalick_HSV);
	fclose (fpharalick_LAB);
	fclose (fpharalick_LUV);
	printf("End");
	while(cvWaitKey(50)!=27);
	return 0;
}
