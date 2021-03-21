/*
 * CoOccurrence.h
 * This file is part of SIAC (Sistema Inteligente para Análise de Cascalho)
 *
 * Copyright (C) 2011 - LCAD - UNESP-Bauru
 *
 * SIAC is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * SIAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SIAC; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, 
 * Boston, MA  02110-1301  USA
 */

/*
Author: Giovanni Chiachia
Modified by: Alan Zanoni Peixinho (alan-peixinho@hotmail.com)
*/


#ifndef COOCCURRENCE_H
#define COOCCURRENCE_H

#include "SymmetricMatrix.h"
// #include <opencv\cv.h>
// #include <opencv\cxcore.h>

// #include <opencv2/opencv.hpp>

using namespace cv;

class CoOccurrenceExtractor
{	
	public:
	
	enum MatrixOrientation{
                MATRIX_0_DEGREES = 0,
                MATRIX_45_DEGREES = 1,
                MATRIX_90_DEGREES = 2,
                MATRIX_135_DEGREES = 3,
				MATRIX_NORMALIZED = 4
	};
	
        CoOccurrenceExtractor(unsigned width, unsigned height, unsigned distance = 1, unsigned quantization = 16);
		~CoOccurrenceExtractor();
	
        void setImage(Mat &image, Mat &roi);
        Mat getImage() const;
	
		double getEnergy(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getContrast(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getCorrelation(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getVariance(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getInverseDifferenceMoment(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getSumAverage(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getSumVariance(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getSumEntropy(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getEntropy(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getDifferenceVariance(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getDifferenceEntropy(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getMeasureOfCorrelation1(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getMeasureOfCorrelation2(MatrixOrientation orientation = MATRIX_NORMALIZED) const;
		double getMaximalCorrelationCoefficient(MatrixOrientation orientation = MATRIX_NORMALIZED);

        private:
        const static int MAX = 5;
        
        typedef unsigned char pixel;

		double *vetRow[MAX];//Px(i)
		double *vetCol[MAX];//Py(j)
		double *vetRowPlusCol[MAX];//Px+y(k)
		double *vetRowMinusCol[MAX];//Px-y(k)
		
		double vetHxy[MAX];//hxy
		double vetHxy1[MAX];//hxy1
		double vetHxy2[MAX];//hxy2
		
		double vetHx[MAX];
		double vetHy[MAX];
		
		double **Q;
		
		double *x;
		double *iy;
		
        SymmetricMatrix<double> *matrix[MAX];
        
        Mat image;
        Mat roi;
        
        unsigned quantization, distance;
        double quantizationCoefficient;

        //retorna valor do pixel referente a [row][col]
        inline pixel pixelAt(unsigned row, unsigned col);
        inline unsigned index(pixel p);
		inline bool validRegion(unsigned row, unsigned col);
        
        inline double sign (double x, double y) const;
        
		void calcCoOccurMatrix(MatrixOrientation orientation);
        void calcRowColVectors(MatrixOrientation orientation);
        void calcHXYValues(MatrixOrientation orientation);
        
        bool hessenberg(double wr[], double wi[]);
        void reductMatrix();
        void balanceMatrix();   
};

#endif//COOCCURRENCE_H
