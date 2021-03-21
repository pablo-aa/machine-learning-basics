/*
Algorithm: Asymmetry property extraction
Author: Roberta Oliveira
Email: roberta.oliveira@fe.up.pt
*/
#include<bits/stdc++.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
//#include <time.h> 

// #include <opencv/cv.h>
// #include <opencv/highgui.h>
// #include <opencv/cxcore.h>
// #include "opencv2/core/core.hpp"
// #include "opencv2/features2d/features2d.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/nonfree/nonfree.hpp"
// #include "opencv2/nonfree/features2d.hpp"
// #include "opencv2/imgproc/imgproc.hpp" //border detector
#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

double geometry(Mat border, int auxmajor);
double moments (Mat border);
double asymmetryIndex(int *bi, int *bj, int x1, int y1, int x2, int y2, int count, int line);
int borderIrregularity(int *bi, int *bj, int x1, int y1, int x2, int y2, int count, int row);

double lesion_area, perimeter, equi_diameter, compactness, circularity, rectangularity, solidity;
double aspect_ratio, eccentricity;
double faverage, fvariance, fdeviation;
int fvalley1, fpeak1, fline1, fvalley2, fpeak2, fline2;

template<typename T>
bool is_infinite( const T &value )
{
    // Since we're a template, it's wise to use std::numeric_limits<T>
    //
    // Note: std::numeric_limits<T>::min() behaves like DBL_MIN, and is the smallest absolute value possible.
    //
 
    T max_value = std::numeric_limits<T>::max();
    T min_value = - max_value;
 
    return ! ( min_value <= value && value <= max_value );
}
 
template<typename T>
bool is_nan( const T &value )
{
    // True if NAN
    return value != value;
}
 
template<typename T>
bool is_valid( const T &value )
{
    return ! is_infinite(value) && ! is_nan(value);
}

//Area features
double geometry(Mat border, int auxmajor)
{
	const double pi = 3.14159265358979323846264338328;
	int i;
	int sizeh, sizew;
	double hull_area, box_area;
	vector<vector<Point>> contours; //Create a vector of the points
	vector<Vec4i> hierarchy;

	//Vector with the conterns
	findContours(border, contours, CV_RETR_EXTERNAL,  CV_CHAIN_APPROX_SIMPLE );
	//printf("Numero de contornos %d\n",contours.size());
	
    // Vector with approximate contours to polygon, bounding box, convex hull, circles
    //vector<vector<Point>>contours_poly(contours.size());
    vector<Rect>box(contours.size());
	vector<vector<Point>>hull(contours.size());
    //vector<Point2f>center(contours.size());
    //vector<float>radius(contours.size());

    for(i=0; i<contours.size(); i++ )
    { 
		//only one countourn
		if (contours.size()==1)
		{
			//Contour approximation - Approximates a polygonal curve(s) with the specified precision.
			//(Mat(contours[i]), contours_poly[i], 3, true);
        
			//Circle
			//minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);

			//Bounding rectangle - Bounding box
			//box[i] = boundingRect(Mat(contours_poly[i]));
			box[i] = boundingRect(Mat(contours[i]));
			sizeh=box[i].height;
			sizew=box[i].width;

			//Convex hull - Finds the convex hull of a point set (circular)
			convexHull(Mat(contours[i]), hull[i], false, true);
		
			//Contourn area
			lesion_area=contourArea(contours[i],false);

			//Countourn perimeter
			perimeter=arcLength(contours[i],true);

			//Equivalent diameter - is the diameter of the circle whose area is same as the contour area.
			equi_diameter=sqrt((4*lesion_area)/pi);

			//Compactness (euclidean distance) - is the ratio of the are of the object to the area of a circle with the same perimeter
			compactness=equi_diameter/auxmajor;

			//circularity or roundness - is the measure of how closely the shape of an object approaches that of a circle.
			circularity=(4*lesion_area*pi)/pow(perimeter,2);

			//Solidity - the ratio of contour area to its convex hull area
			hull_area=contourArea(hull[i],false);
			solidity=lesion_area/hull_area;

			//rectangularity (extent) - the ratio of contour area to bounding rectangle area (Bounding box).
			box_area=sizew*sizeh;
			rectangularity = lesion_area/box_area;

			// Draw polygonal contour + bonding rects
			 Mat drawing = border.clone();
			 cvtColor(drawing, drawing, CV_GRAY2BGR);
			 //for(i=0; i<contours.size(); i++)
			 //{
				 rectangle(drawing, box[i].tl(), box[i].br(), Scalar(255,0,0), 2, 8, 0);          
				 drawContours( drawing, hull, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point());
				 //drawContours(drawing, contours_poly, i, Scalar(0,255,0), 1, 8, vector<Vec4i>(), 0, Point());
				 //circle(drawing, center[i], (int)radius[i], Scalar(255,255,255), 2, 8, 0);
			 //}
			 //imshow("Result", drawing);
		}
		//more than one countourn
		else if (contours.size()>1)
		{
			i=(contours.size()-1);
			//Contour approximation - Approximates a polygonal curve(s) with the specified precision.
			//(Mat(contours[i]), contours_poly[i], 3, true);
        
			//Circle
			//minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);

			//Bounding rectangle - Bounding box
			//box[i] = boundingRect(Mat(contours_poly[i]));
			box[i] = boundingRect(Mat(contours[i]));
			sizeh=box[i].height;
			sizew=box[i].width;

			//Convex hull - Finds the convex hull of a point set (circular)
			convexHull(Mat(contours[i]), hull[i], false, true);
		
			//Contourn area
			lesion_area=contourArea(contours[i],false);

			//Countourn perimeter
			perimeter=arcLength(contours[i],true);

			//Equivalent diameter - is the diameter of the circle whose area is same as the contour area.
			equi_diameter=sqrt((4*lesion_area)/pi);

			//Compactness (euclidean distance) - is the ratio of the are of the object to the area of a circle with the same perimeter
			compactness=equi_diameter/auxmajor;

			//circularity or roundness - is the measure of how closely the shape of an object approaches that of a circle.
			circularity=(4*lesion_area*pi)/pow(perimeter,2);

			//Solidity - the ratio of contour area to its convex hull area
			hull_area=contourArea(hull[i],false);
			solidity=lesion_area/hull_area;

			//rectangularity (extent) - the ratio of contour area to bounding rectangle area (Bounding box).
			box_area=sizew*sizeh;
			rectangularity = lesion_area/box_area;


			// Draw polygonal contour + bonding rects
			 Mat drawing = border.clone();
			 cvtColor(drawing, drawing, CV_GRAY2BGR);
			 //for(i=0; i<contours.size(); i++)
			 //{
				 rectangle(drawing, box[i].tl(), box[i].br(), Scalar(255,0,0), 2, 8, 0);          
				 drawContours( drawing, hull, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point());
				 //drawContours(drawing, contours_poly, i, Scalar(0,255,0), 1, 8, vector<Vec4i>(), 0, Point());
				 //circle(drawing, center[i], (int)radius[i], Scalar(255,255,255), 2, 8, 0);
			 //}
			 //imshow("Result", drawing);

			 i=contours.size();
		}
     }

	 return 0;
}

//Moments features
double moments (Mat border)
{
	double spatial[10], central[7];
	double major_axis, minor_axis;
	int cx, cy;

	//Spatial moments - only one contourn
	Moments mom1 = moments(border,true);
	spatial[0] = mom1.m00;
	spatial[1] = mom1.m10;
	spatial[2] = mom1.m01;
	spatial[3] = mom1.m20;
	spatial[4] = mom1.m11;
	spatial[5] = mom1.m02;
	spatial[6] = mom1.m30;
	spatial[7] = mom1.m21;
	spatial[8] = mom1.m12;
	spatial[9] = mom1.m03;

	//Mass center or Centroid, it is used fot central moments
	cx = int(mom1.m10/mom1.m00);
    cy = int(mom1.m01/mom1.m00);

	//Central moments - calculate from the centroid
	Moments mom2 = moments(border,true);
	central[0] = mom2.mu20;
	central[1] = mom2.mu11;
	central[2] = mom2.mu02;
	central[3] = mom2.mu30;
	central[4] = mom2.mu21;
	central[5] = mom2.mu12;
	central[6] = mom2.mu03;
	
	//Lenght of the major axis of the object
	//double aux_major1=((mom2.mu02-mom2.mu20)*(mom2.mu02-mom2.mu20))+4*mom2.mu11;
	//double aux_major2=pow(aux_major1,0.5);
	//double aux_major3=(mom2.mu02+mom2.mu20)+aux_major2;
	//double aux_major4 = pow((8*aux_major3),0.5);
	//major_axis=pow(8*(mom2.mu02+mom2.mu20+(pow((mom2.mu02-mom2.mu20)*(mom2.mu02-mom2.mu20)+4*mom2.mu11,0.5))),0.5);
	major_axis=sqrt(8*(mom2.mu02+mom2.mu20+(sqrt((mom2.mu02-mom2.mu20)*(mom2.mu02-mom2.mu20)+4*mom2.mu11))));
	//printf("major_axis %f\n", major_axis);
	//Lenght of the minor axis of the object
	//minor_axis=pow(8*(mom2.mu02+mom2.mu20-(pow((mom2.mu02-mom2.mu20)*(mom2.mu02-mom2.mu20)+4*mom2.mu11,0.5))),0.5);
	minor_axis=sqrt(8*(mom2.mu02+mom2.mu20-(sqrt((mom2.mu02-mom2.mu20)*(mom2.mu02-mom2.mu20)+4*mom2.mu11))));
	//printf("minor_axis %f\n", minor_axis);
	
	if(is_valid(major_axis) || is_valid(minor_axis))
	{
		//Determine the aspect ratio (lengthening index) - ratio of the lenght of the major axis to the legnght of the minor axis
		aspect_ratio=major_axis/minor_axis;
		//printf("aspect_ratio %f\n", aspect_ratio);
		//Eccentricity (a measure of elongation)
		eccentricity=(((mom2.mu02-mom2.mu20)*(mom2.mu02-mom2.mu20))+4*mom2.mu11)/((mom2.mu02+mom2.mu02)*(mom2.mu02+mom2.mu02));
		//printf("eccentricity %f\n", eccentricity);
	}
	else
	{
		aspect_ratio=0;
		eccentricity=0;
	}

	//cout << "Value of a: " << a << " " << is_valid(a) << " " << (is_nan(a) ? " nan " : "") << (is_infinite(a) ? " infinite " : "") << "n";

	return 0;
}

//Asymmetry property extraction
double asymmetryIndex (int *bi, int *bj, int x1, int y1, int x2, int y2, int count, int line)
{
	int *a=NULL,*b=NULL, *va=NULL, *vb=NULL;
	double *ratioperp=NULL;
	int maxpoint, point, xii, yjj, guard1, guard2, perp, numperp, f, d, i, m, bx1, by1, bx2, by2;
	double xaux, yaux, s, c1, c0, c2, c00, p1, p2, distance1, distance2, num,
	       sum, value, average, variance, deviation;

	if (a!=NULL)
	{
		free (a);
		a=NULL;
	}
	if (b!=NULL)
	{
		free (b);
		b=NULL;
	}

	if (va!=NULL)
	{
		free (va);
		va=NULL;
	}
	if (vb!=NULL)
	{
		free (vb);
		vb=NULL;
	}
	if (ratioperp!=NULL)
	{
		free (ratioperp);
		ratioperp=NULL;
	}

	//Find the equation of straight line between two points
    //Determine the points of the longest diagonal
	maxpoint=(x2-x1)*10; //Because xaux+=0.1
	a=(int *)malloc(maxpoint * sizeof(int));
	b=(int *)malloc(maxpoint * sizeof(int));
	for (int i=0; i<maxpoint; i++)
	{
		a[i]=0;
		b[i]=0;
	}

	point=0;
	for (xaux=x1; xaux<=x2; xaux+=0.1)
	//for (xaux=x1; xaux<=x2; xaux++)
	{
        yaux=(xaux*(y1-y2)+x1*y2-y1*x2)/(x1-x2);
        xii=(_int16)(xaux);
        yjj=(_int16)(yaux);
        //Vector with the points of the longest diagonal 
        point++;
        a[point]=xii;
        b[point]=yjj;
	}
  
	//Find the number of perpendicular lines from the longest diagonal
	//d (point) - number of points of the diagonal, 
	//f (count) - number of points of the countorn, 
	//x and y - first and second point of the longest diagonal, 
	//bi and bj (vectors) - points of the countorn, 
	//a and b (vectors) - points of the diagonal
  
	va=(int *)malloc(point * sizeof(int));
	vb=(int *)malloc(point * sizeof(int));
	for (int i=0; i<point; i++)
	{
		va[i]=0;
		vb[i]=0;
	}
	perp=0;
	//not use the first and last points of the diagonal
	for (d=11; d<point-10; d++)//Because xaux+=0.1 (10 points)
	{
		guard1 = 0;
		for (f=1; f<=count; f++)
		{
			c0 = ((y1-y2)/(x1-x2));
			if(c0 < 1)
			{
				c0 = 0;
				c1=c0*(bi[f]-a[d])+b[d]-bj[f];
			}
			else
		    	c1=(-1/c0)*(bi[f]-a[d])+b[d]-bj[f];
			if ((c1==0)&&(guard1==1))
			{
				guard1=2;
				perp++;
				va[perp]=a[d];
				vb[perp]=b[d];				
			}
			else if ((c1==0)&&(guard1==0))
				guard1=1;
		}    
	}

	//Calculate the number expected perpendicular lines (dintance between the points)  - e.g. 10 samples (perpendicular lines) 
	
	ratioperp=(double *)malloc(perp * sizeof(double));
	for (i=0; i<perp; i++)
		ratioperp[i]=0;

	numperp=0;
	if (perp>line)
		num=perp/line;
	else
		num=1;
	for (s=num; s<=perp; s+=num)
	{
		d=(_int16)(s);
		guard2=0;
		for (f=1; f<=count; f++)
		{
			c00 = ((y1-y2)/(x1-x2));
			if(c00 < 1)
			{
				c00 = 0;
				c2=c00*(bi[f]-va[d])+vb[d]-bj[f];
			}
			else
		    	c2=(-1/c00)*(bi[f]-va[d])+vb[d]-bj[f];

			if ((c2==0)&&(guard2==1))
			{
				guard2=2;
				numperp++;
				bx2=bi[f];
				by2=bj[f];
				p1=(double)(va[d]);
				p2=(double)(vb[d]);
				distance1=sqrt(pow(p1-bx1,2)+pow(p2-by1,2));
				distance2=sqrt(pow(p1-bx2,2)+pow(p2-by2,2));
                
				//Calculate the ratio between the two distances (minor/major - between 0 (asymmetric) and 1 (symmetric))
				if (distance1>=distance2)
					ratioperp[numperp]=distance2/distance1;
				else
					ratioperp[numperp]=distance1/distance2;
				
			}
			if ((c2==0) && (guard2==0))
			{
				guard2=1;
				bx1=bi[f];
				by1=bj[f];
			}
		}
	}
	
	//Calculate the standard deviation of the all samples (10 perpendiculars or all perpendiculars)
	sum=0, value=0;
	for (m=1; m<=numperp; m++)
		sum+=ratioperp[m];
	
	if (numperp>0)
		average=sum/numperp;
	else
		average=0;
	
	for (int m=1; m<=numperp; m++)
		value+=pow(ratioperp[m]-average,2);
	
	if (numperp>1)
		variance=value/(numperp-1);
	else
		variance=0;
	
	if (numperp>1)
		deviation=sqrt(variance);
	else
		deviation=0;
	
	memcpy(&faverage, &average, sizeof(double));
	memcpy(&fvariance, &variance, sizeof(double));
	memcpy(&fdeviation, &deviation, sizeof(double));

	return 0;
}

//Border property extraction
int borderIrregularity (int *bi, int *bj, int x1, int y1, int x2, int y2, int count, int row)
{
	int *xx=NULL,*yy=NULL;
    int mpx, mpy, contourn, inverse, distance2,
		sample1, n, p1, p2, p3, valley1, peak1, line1;
	double px, py, p;

	if (xx!=NULL)
	{
		free (xx);
		xx=NULL;
	}
	if (yy!=NULL)
	{
		free (yy);
		yy=NULL;
	}

	//One-dimensional border
	xx=(int *)malloc(count * sizeof(int));
	yy=(int *)malloc(count * sizeof(int));
	for (int i=0; i<count; i++)
	{
		xx[i]=0;
		yy[i]=0;
	}
	
	//Find the central point from the greatest distance
	mpx=(unsigned int)((x1 + x2)/2);
    mpy=(unsigned int)((y1 + y2)/2);
	px=(double)mpx;
    py=(double)mpy;

    //Find the distance from the central point for each border point    	
    for (contourn=1; contourn<=count; contourn++)
	{
        distance2=(unsigned int)(sqrt(pow(bi[contourn]-mpx,2)+pow(bj[contourn]-mpy,2)));
		inverse=row-distance2;
        xx[contourn]=inverse; //lines
        yy[contourn]=contourn; //columns++
	}

	//Calculate the number of peaks and valleys based on large irregularities
    //Vectorial product (n with interval of 15 pixels)
    sample1=0, valley1=0, peak1=0, line1=0;
	for (n=16; n<=count-15; n+=15)
	{
        p1=n-15;
        p2=n;
        p3=n+15;
        p=(xx[p2]-xx[p1])*(yy[p3]-yy[p1])-(yy[p2]-yy[p1])*(xx[p3]-xx[p1]);
        sample1++;
        if (p<0)
            valley1++;
        else if (p>0)
            peak1++;
        else if (p==0)
            line1++;
	}
	
	//Calculate the number of peaks and valleys based on small irregularities
    //Inflexion point (using interval of 4 pixels)
    int sample2=0, i=5;
    int pc, pe1, pe2, pe3, pe4, pd1, pd2, pd3, pd4;
	int peso, pesoesq, pesodir, pesoesq1, pesoesq2, pesoesq3, pesoesq4, pesodir1, pesodir2, pesodir3, pesodir4; 
	int valley2=0, peak2=0, line2=0;
    
    while (i<count-4){
        pc=i;
        pe1=i-1;
        pe2=i-2;
        pe3=i-3;
        pe4=i-4;
        pd1=i+1;
        pd2=i+2;
        pd3=i+3;
        pd4=i+4;
                
        //Assign and accumulate the weights 
		//Each neighbour pixel that is below the pixels under analysis (systems of coordinates - vertical axis) receives weight = -1, above receives weight = 1. 
		//Otherwise each neighbour pixel receives 0.
        if (xx[pe1]>xx[pc])
			pesoesq1=1;
        else if (xx[pe1]<xx[pc])
            pesoesq1=-1;
        else
            pesoesq1=0;

        if (xx[pe2]>xx[pc])
            pesoesq2 = 1;
        else if (xx[pe2]<xx[pc])
            pesoesq2=-1;
        else
            pesoesq2=0;

        if (xx[pe3]>xx[pc])
            pesoesq3=1;
        else if (xx[pe3]<xx[pc])
            pesoesq3=-1;
        else
            pesoesq3=0;

        if (xx[pe4]>xx[pc])
            pesoesq4=1;
        else if (xx[pe4]<xx[pc])
            pesoesq4=-1;
        else
            pesoesq4=0;

        if (xx[pd1]>xx[pc])
            pesodir1=1;
        else if (xx[pd1]<xx[pc])
            pesodir1=-1;
        else
            pesodir1=0;

        if (xx[pd2]>xx[pc])
            pesodir2=1;
        else if (xx[pd2]<xx[pc])
            pesodir2=-1;
        else
            pesodir2=0;

        if (xx[pd3]>xx[pc])
            pesodir3=1;
        else if (xx[pd3]<xx[pc])
            pesodir3=-1;
        else
            pesodir3=0;

        if (xx[pd4]>xx[pc])
            pesodir4=1;
        else if (xx[pd4]<xx[pc])
            pesodir4=-1;
        else
            pesodir4=0;
        
        pesoesq=pesoesq1+pesoesq2+pesoesq3+pesoesq4;
        pesodir=pesodir1+pesodir2+pesodir3+pesodir4;
       
        //Analyse if the point is an inflexion (possible inflection point)
        if ((pesoesq>=2 || pesoesq<=-2) && (pesodir>=2 || pesodir<=-2))
		{
            //Number of possible inflexions
            sample2+=1;
            //Analyse if the point of inflextion is a peak or a valley
            peso=pesoesq+pesodir;
            if (peso>0)
                peak2++;
            else if (peso<0)
                valley2++;
			else // it is not an inflection point
				line2++;

            i+=4;
		}
        else
            i+=1;
	}
	memcpy(&fpeak1, &peak1, sizeof(int));
	memcpy(&fvalley1, &valley1, sizeof(int));
	memcpy(&fline1, &line1, sizeof(int));

	memcpy(&fpeak2, &peak2, sizeof(int));
	memcpy(&fvalley2, &valley2, sizeof(int));
	memcpy(&fline2, &line2, sizeof(int));
	
	return 0;
}