#pragma warning(disable:4996)

#include "CoOccurrence.h"

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

const double LOG10_2 = log10(2);
const double RADIX = 2;
const double EPSILON = std::numeric_limits<double>::epsilon();//aproximadamente 2*(10^⁻16)

inline unsigned CoOccurrenceExtractor::index(CoOccurrenceExtractor::pixel p)
{
    return quantizationCoefficient*p;
}

inline CoOccurrenceExtractor::pixel CoOccurrenceExtractor::pixelAt(unsigned row, unsigned col)
{
	return image.at<CoOccurrenceExtractor::pixel>(row, col);
}

inline bool CoOccurrenceExtractor::validRegion(unsigned row, unsigned col)
{
    return roi.at<CoOccurrenceExtractor::pixel>(row, col)>0? true : false;
}

inline double CoOccurrenceExtractor::sign(double x, double y) const
{
	return (y<0 ? -fabs(x) : fabs(x));
}

CoOccurrenceExtractor::CoOccurrenceExtractor(unsigned width, unsigned height, unsigned distance, unsigned quantization)
        :distance(distance), quantization(quantization)
{
    quantizationCoefficient = double(quantization)/256;

    for(unsigned i=0; i<5; ++i)
    {
        matrix[i] = new SymmetricMatrix<double>(quantization);
        
        vetRow[i] = new double[quantization];
        vetCol[i] = new double[quantization];
        
        vetRowPlusCol[i] = new double[2 * quantization];
		vetRowMinusCol[i] = new double[2 * quantization];
    }
    
    Q = new double*[quantization];

    
    for(unsigned i=0; i<quantization; ++i)
    {
    	Q[i] = new double[quantization];
    }
    
    x = new double[quantization];
  	iy = new double[quantization];
  	

    image = cv::Mat( cvSize(width, height), IPL_DEPTH_8U, 1);

    //if(img!=NULL)
		//setImage(img);

}

CoOccurrenceExtractor::~CoOccurrenceExtractor()
{
    //cvReleaseImage(&image);
    for(unsigned i=0; i<5; ++i)
        delete matrix[i];
        
    for(unsigned i=0; i<quantization; ++i)
  		delete[] (Q[i]);
  		
	delete []Q;	
	delete []x;
	delete []iy;        
    
}

void CoOccurrenceExtractor::setImage(Mat& img, Mat& roi)
{
	SymmetricMatrix<double>* m;
	
	double normFactor = 4 * // number of orientations.
                      2 * // the co-occurrence associative property.
                      ( quantization - 1 ) - // the reduced dimension.
                      quantization; // accounts for the not repeated diagonal.
					  
	if(img.channels() == 1)//nao esta em escala de cinza
		image = img.clone();
	//else
		//cvtColor( img, image, CV_RGB2GRAY);//esta em escala de cinza
	
	this->roi = roi.clone();

	//calcula matrizes de co-ocorrencia
    for(unsigned i=0; i<MAX; ++i)//zera matrizes
        std::fill(matrix[i]->begin(), matrix[i]->end(), 0);

	for(unsigned i = 0; i<MAX; ++i)
	{
		MatrixOrientation orientation = MatrixOrientation(i);
		calcCoOccurMatrix(orientation);
    	calcRowColVectors(orientation);
    	calcHXYValues(orientation);
    }
    
    //calcula matriz normalizada, atraves das quatro orientacoes	
	SymmetricMatrix<double>::iterator i0 = matrix[MATRIX_0_DEGREES]->begin(),
	i45 = matrix[MATRIX_45_DEGREES]->begin(),
	i90 = matrix[MATRIX_90_DEGREES]->begin(),
	i135 = matrix[MATRIX_135_DEGREES]->begin();
	
 	for(SymmetricMatrix<double>::iterator i = matrix[MATRIX_NORMALIZED]->begin(); i!=matrix[4]->end(); ++i)
	{
		*i =   ( (*i0) / normFactor /   quantization ) + ( (*i45) / normFactor / ( quantization - 1 ) ) +
               ( (*i90) / normFactor /   quantization ) + ( (*i135) / normFactor / ( quantization - 1 ) );
		
		++i0;
		++i45;
		++i90;
		++i135;
	}
	
	calcRowColVectors(MATRIX_NORMALIZED);
    calcHXYValues(MATRIX_NORMALIZED);
    
}

void CoOccurrenceExtractor::calcCoOccurMatrix(MatrixOrientation orientation)
{
    SymmetricMatrix<double> *m = matrix[orientation];

    unsigned dxA, dxB, dyA, dyB;
    unsigned xMax = image.size().width;
    unsigned yMax = image.size().height;

    pixel pixelA, pixelB;

    unsigned indexedA, indexedB;

    dxA = dxB = dyA = dyB = 0;

    switch(orientation)
    {
         case MATRIX_0_DEGREES:
            dxB += distance;
            xMax -= distance;
            break;
         case MATRIX_45_DEGREES:
            dyA += distance;
            dxB += distance;
            xMax -= distance;
            yMax -= distance;
            break;
         case MATRIX_90_DEGREES:
            dyB += distance;
            yMax -= distance;
            break;
         case MATRIX_135_DEGREES:
            dxB += distance;
            dyB += distance;
            xMax -= distance;
            yMax -= distance;
            break;
         default:
            return;
    }
    
    for( unsigned i = 0; i < yMax; ++i )
    {
      for( unsigned j = 0; j < xMax; ++j )
      {
		 //Verificar somente os pontos do ROI
		 //if(!validRegion(i+dyA, j+dxA) || !validRegion(i+dyB, j+dxB))
		//	continue;
			
         //cont = cont + 1;
		 //Access the image's pixels
         pixelA = pixelAt(i+dyA, j+dxA);
         pixelB = pixelAt(i+dyB, j+dxB);
		
		//Quantize the array accordiquantization to quantization
		indexedA = index(pixelA);
		indexedB = index(pixelB);

		//Add co-occurrence to the array
		if(indexedA==indexedB)//diagonal principal nao possui espelho, portanto a incremento duas vezes
			  m->at(indexedA, indexedB)+=2;
		else
			  m->at(indexedA, indexedB)++;	
		 	 
      }
    }
	

    //sum of matrix elements (includiquantization mirrors) 
    double matrixSum = 2*yMax*xMax;

    //normalize matrix in [0, 1]
    for(SymmetricMatrix<double>::iterator it = m->begin(); it!=m->end(); ++it)
        (*it)/=matrixSum;
	
}

void CoOccurrenceExtractor::calcRowColVectors(MatrixOrientation orientation)
{
	double element;
	SymmetricMatrix<double> *m = matrix[orientation];

	double *row = vetRow[orientation];
	double *col = vetCol[orientation];
	
	double *rowPlusCol = vetRowPlusCol[orientation];
	double *rowMinusCol = vetRowMinusCol[orientation];	

	for(unsigned i=0; i<quantization; ++i)
		row[i] = col[i] = 0;
		
	for(unsigned i=0; i < 2 * quantization; ++i)
		rowPlusCol[i] = rowMinusCol[i] = 0;
	
	for(unsigned i=0; i<quantization; ++i)
	{
		for(unsigned j=0; j<quantization; ++j)
		{
			element = m->at(i, j);
			
			row[i] += element;
			col[j] += element;
			
			rowPlusCol[i+j] +=element;
			rowMinusCol[std::abs(int(i-j))] += element;
		}
	}
}

void CoOccurrenceExtractor::calcHXYValues(MatrixOrientation orientation)
{
	double element;
	
	double *row = vetRow[orientation];
	double *col = vetCol[orientation];
	
	SymmetricMatrix<double> *m = matrix[orientation];
	
	
	vetHxy[orientation] = vetHxy1[orientation] = vetHxy2[orientation] = vetHx[orientation] = vetHy[orientation] = 0;	
	for(unsigned i=0; i<quantization; ++i)
	{
		for(unsigned j=0; j<quantization; ++j)
		{
			element = m->at(i, j);
			vetHxy1[orientation] -= element * log10 (row[i] * col[j] + EPSILON)/LOG10_2;
  			vetHxy2[orientation] -= row[i] * col[j] * log10 (row[i] * col[j] + EPSILON)/LOG10_2;
  			vetHxy[orientation] -= element * log10 (element + EPSILON)/LOG10_2;
  		}
  	}
  	
  	for (unsigned i = 0; i < quantization; ++i)
  	{
		vetHx[orientation] -= row[i] * log10 (row[i] + EPSILON)/LOG10_2;
		vetHy[orientation] -= col[i] * log10 (col[i] + EPSILON)/LOG10_2;
	}
}


/************************************************************************************************************************/
/*							HARALICK FEATURES											*/
/************************************************************************************************************************/

//energia, ou segundo momento aquantizationular
double CoOccurrenceExtractor::getEnergy(MatrixOrientation orientation) const
{
   double sum = 0, element;

   SymmetricMatrix<double> *m = matrix[orientation];

   for ( unsigned int i = 0; i < quantization; ++i )
   {
      for ( unsigned int j = 0; j < quantization; ++j )
      {
         element = m->at(i, j);
         sum += element * element;  //sum += pow( m->at(i, j), ( double) 2);
      }
   }
   return sum;
}

double CoOccurrenceExtractor::getContrast(MatrixOrientation orientation) const
{
   double sum = 0;

   SymmetricMatrix<double> *m = matrix[orientation];

   for ( unsigned int i = 0; i < quantization; ++i )
   {
      for ( unsigned int j = 0; j < quantization; ++j )
      {
         sum += m->at(i, j) * (i-j) * (i-j);  //sum += m->at(i, j)* pow((i-j), 2)
      }
   }
   
   return sum;
}


double CoOccurrenceExtractor::getCorrelation(MatrixOrientation orientation) const
{
	double element;
	double meanX = 0, meanY = 0;
	double stdDevX = 0, stdDevY = 0;
	
	double *row = vetRow[orientation];
	
	SymmetricMatrix<double> *m = matrix[orientation];
	
	double sum = 0;
	
	for(unsigned i=0; i<quantization; ++i)//calcula media
	{
		meanX += row[i]*i;
    	stdDevX += row[i]*i*i;//soma dos quadrados
	}
	
	meanY = meanX;
  	stdDevY = stdDevX;
  	stdDevX = std::sqrt(std::fabs(stdDevX - (meanX * meanX)));
  	stdDevY = stdDevX;	
	
	for(unsigned i = 0; i < quantization; ++i)
		for(unsigned j = 0; j < quantization; ++j)
			sum += i*j*m->at(i, j);
			
	return (sum - meanX*meanY) / (stdDevX * stdDevY);
}

double CoOccurrenceExtractor::getVariance(MatrixOrientation orientation) const
{
	double sum = 0, mean = 0;
	SymmetricMatrix<double> *m = matrix[orientation];
	
	for(unsigned i=0; i<quantization; ++i)
	{
		for(unsigned j=0; j<quantization; ++j)
		{
			mean += i * m->at(i, j);//eh calculada a media dos pixel, e nao da matriz de co-ocorrencia
		}
	}	
	
	for(unsigned i=0; i<quantization; ++i)
	{
		for(unsigned j=0; j<quantization; ++j)
		{
			sum += (i - mean) * (i - mean) * m->at(i, j);
		}
	}
	return sum;
}

double CoOccurrenceExtractor::getInverseDifferenceMoment(MatrixOrientation orientation) const
{
  double sum = 0;
  SymmetricMatrix<double> *m = matrix[orientation];

  for (unsigned i = 0; i < quantization; ++i)
    for (unsigned j = 0; j < quantization; ++j)
      sum += m->at(i, j) / (1 + (i - j) * (i - j));

  return sum;
}

double CoOccurrenceExtractor::getSumAverage(MatrixOrientation orientation) const
{
	double sum = 0;
	SymmetricMatrix<double> *m = matrix[orientation];
	
	double *rowPlusCol = vetRowPlusCol[orientation];
	
	for (unsigned i = 0; i <= (2 * quantization - 2); ++i)
	{
    	sum += i * rowPlusCol[i];
    }
    
    return sum;
}

double CoOccurrenceExtractor::getSumVariance(MatrixOrientation orientation) const
{
	double sum = 0;
	SymmetricMatrix<double> *m = matrix[orientation];
	
	double feature8 = getSumEntropy(orientation);

	double *rowPlusCol = vetRowPlusCol[orientation];
	

	for (unsigned i = 0; i <= (2 * quantization - 2); ++i)
	{
		sum += (i - feature8) * (i - feature8) * rowPlusCol[i];
	}

	return sum;
}

double CoOccurrenceExtractor::getSumEntropy(MatrixOrientation orientation) const
{
	SymmetricMatrix<double> *m = matrix[orientation];
	double sum = 0;
	double *rowPlusCol = vetRowPlusCol[orientation];

	for (unsigned i = 0; i <= (2 *quantization - 2); ++i)
	{
		sum -= rowPlusCol[i] * std::log10 (rowPlusCol[i] + EPSILON)/LOG10_2; ;
	}
	
	return sum;
}

double CoOccurrenceExtractor::getEntropy(MatrixOrientation orientation) const
{
   double sum = 0, element;

   SymmetricMatrix<double> *m = matrix[orientation];

   for ( unsigned int i = 0; i < quantization; i++ )
   {
      for ( unsigned int j = 0; j < quantization; j++ )
      {
         element = m->at(i, j);
         sum += element * ( std::log10(element + EPSILON)/LOG10_2);
         //sum+=m->at(i, j)*(log10(m->at(i, j))/log10(2))
      }
   }
   return sum;
}

double CoOccurrenceExtractor::getDifferenceVariance(MatrixOrientation orientation) const
{
  long double sumSqr = 0, var = 0, sum = 0;
  
  double *rowMinusCol = vetRowMinusCol[orientation];

  for (unsigned i = 0; i < quantization; ++i)
  {
    sum += i * rowMinusCol[i] ;
    sumSqr += i * i * rowMinusCol[i] ;
  }
  
  return (sumSqr - sum*sum);
}

double CoOccurrenceExtractor::getDifferenceEntropy(MatrixOrientation orientation) const
{
  double sum = 0, sum_sqr = 0, var = 0;
  
  double *rowMinusCol = vetRowMinusCol[orientation];

  for (unsigned i = 0; i < quantization; ++i)
  {
  	sum += rowMinusCol[i] * log10 (rowMinusCol[i] + EPSILON)/LOG10_2;
  }

  return -sum;
}

double CoOccurrenceExtractor::getMeasureOfCorrelation1(MatrixOrientation orientation) const
{
	double hxy1, hxy, hx, hy;
	
	hxy1 = vetHxy1[orientation];
	hxy = vetHxy[orientation];
	hx = vetHx[orientation];
	hy = vetHy[orientation];
	
	return (hxy - hxy1) / (hx > hy ? hx : hy);
}

double CoOccurrenceExtractor::getMeasureOfCorrelation2(MatrixOrientation orientation) const
{
	double hxy2, hxy;
	
	hxy2 = vetHxy2[orientation];
	hxy = vetHxy[orientation];
	
	return std::sqrt (std::fabs(1 - std::exp (-2.0 * (hxy2 - hxy))));
}

double CoOccurrenceExtractor::getMaximalCorrelationCoefficient(MatrixOrientation orientation)
{

  double *row = vetRow[orientation];
  double *col = vetCol[orientation];
  
  SymmetricMatrix<double> *m = matrix[orientation];
  
  double tmp;
  
  row = vetRow[orientation];
  col = vetCol[orientation];

	for (unsigned i = 0; i < quantization; ++i)
	{
		for (unsigned j = 0; j < quantization; ++j)
		{
			Q[i ][j ] = 0;
			for (unsigned k = 0; k < quantization; ++k)
				Q[i ][j ] += m->at(i, k) * m->at(j, k) / row[i] / col[k];
		}
	}

	balanceMatrix();
	
	reductMatrix();
  
	if (!hessenberg (x, iy))
	{
		return 0;
	}

	//procura o segundo maior autovalor
	double first, second;
	first = x[0];
	second = x[1];
  	
  	for (unsigned i = 2; i < quantization; ++i)
  	{
  		if(x[i]>first)
  		{
  			second = first;
  			first = x[i];
  		}
  		else if(x[i]>second)
  			second = x[i];
  	}

	return std::sqrt(second);
}


bool CoOccurrenceExtractor::hessenberg(double wr[], double wi[])
{
  int nn, m, l, k, j, its, i, mmin;
  double z, y, x, w, v, u, t, s, r, q, p, anorm;

  anorm = std::fabs (Q[0][0]);//---
  
  for (i = 2; i <= quantization-1; ++i)
    for (j = (i - 1); j <= quantization-1; ++j)
      anorm += fabs (Q[i-1][j-1]);//---
  nn = quantization-1;
  t = 0;
  while (nn >= 1)
  {
    its = 0;
    do
    {
      for (l = nn; l >= 2; l--)
      {
		s = fabs (Q[l - 2][l - 2]) + fabs (Q[l-1][l-1]);//---
		if (s == 0)
		  s = anorm;
		if ((double) (std::fabs (Q[l-1][l - 2]) + s) == s)//---
		  break;
      }
      x = Q[nn-1][nn-1];
      if (l == nn)
      {
		wr[nn-1] = x + t;//---
		wi[nn-1] = 0;//---
		--nn;
      }
      else
      {
	y = Q[nn - 2][nn - 2];//---
	w = Q[nn-1][nn - 2] * Q[nn - 2][nn-1];//---
	if (l == (nn - 1))
	{
	  p = 0.5 * (y - x);
	  q = p * p + w;
	  z = std::sqrt (std::fabs (q));
	  x += t;
	  if (q >= 0)
	  {
	    z = p + sign (z, p); 
	    wr[nn - 2] = wr[nn-1] = x + z;//---
	    if (z)
	      wr[nn-1] = x - w / z;//---
	    wi[nn - 2] = wi[nn-1] = 0;//---
	  }
	  else
	  {
	    wr[nn - 2] = wr[nn-1] = x + p;//---
	    wi[nn - 2] = -(wi[nn-1] = z);//---
	  }
	  nn -= 2;
	}
	else
	{
	  if (its == 30)
	  {
	     return false;
	  }			
	  if (its == 10 || its == 20)
	  {
	    t += x;
	    for (i = 1; i <= nn; i++)
	      Q[i-1][i-1] -= x;
	    s = fabs (Q[nn-1][nn - 2]) + fabs (Q[nn - 2][nn - 3]);
	    y = x = 0.75 * s;
	    w = -0.4375 * s * s;
	  }
	  ++its;
	  for (m = (nn - 2); m >= l; m--)
	  {
	    z = Q[m-1][m-1];
	    r = x - z;
	    s = y - z;
	    p = (r * s - w) / Q[m ][m-1] + Q[m-1][m ];
	    q = Q[m ][m ] - z - r - s;
	    r = Q[m +1][m ];
	    s = fabs (p) + fabs (q) + fabs (r);
	    p /= s;
	    q /= s;
	    r /= s;
	    if (m == l)
	      break;
	    u = fabs (Q[m-1][m - 2]) * (fabs (q) + fabs (r));
	    v = fabs (p) * (fabs (Q[m - 2][m - 2]) + 
			    fabs (z) + fabs (Q[m ][m ]));
	    if ((float) (u + v) == v)
	      break;
	  }
	  for (i = m + 2; i <= nn; i++)
	  {
	    Q[i-1][i - 3] = 0;

	    if (i != (m + 2))
	      Q[i-1][i - 4] = 0;
	  }
	  for (k = m; k <= nn - 1; k++)
	  {
	    if (k != m)
	    {
	      p = Q[k-1][k - 2];
	      q = Q[k ][k - 2];//---
	      r = 0;
	      if (k != (nn - 1))
			r = Q[k + 1][k - 2];//---
	      if (x = fabs (p) + fabs (q) + fabs (r))
	      {
			p /= x;
			q /= x;
			r /= x;
	      }
	    }
	    if (s = sign (sqrt (p * p + q * q + r * r), p)) 
	    {
	      if (k == m)
	      {
			if (l != m)
			  Q[k-1][k - 2] = -Q[k-1][k - 2];//---
	      }
	      else
			Q[k-1][k - 2] = -s * x;
	      p += s;
	      x = p / s;
	      y = q / s;
	      z = r / s;
	      q /= p;
	      r /= p;
	      for (j = k; j <= nn; j++)
	      {
			p = Q[k-1][j-1] + q * Q[k ][j-1];
			if (k != (nn - 1))
			{
			  p += r * Q[k + 1][j-1];
			  Q[k + 1][j-1] -= p * z;
			}
			Q[k ][j-1] -= p * y;
			Q[k-1][j-1] -= p * x;
	      }
	      mmin = nn < k + 3 ? nn : k + 3;
	      for (i = l; i <= mmin; i++)
	      {
			p = x * Q[i-1][k-1] + y * Q[i-1][k ];
			if (k != (nn - 1))
			{
			  p += z * Q[i-1][k + 1];
			  Q[i-1][k + 1] -= p * r;
			}
			Q[i-1][k ] -= p * q;
			Q[i-1][k-1] -= p;
	      }
	    }
	  }
	}
      }
    } while (l < nn - 1);
  }
	return true;
}

void CoOccurrenceExtractor::reductMatrix()
{
  int m, j, i;
  float y, x;

  for (m = 2; m < quantization-1; m++)
  {
    x = 0.0;
    i = m;
    for (j = m; j <= quantization-1; j++)
    {
      if (fabs (Q[j-1][m - 2]) > fabs (x))
      {
	x = Q[j-1][m - 2];
	i = j;
      }
    }
    if (i != m)
    {
      for (j = m - 1; j <= quantization-1; j++)
	std::swap (Q[i-1][j-1], Q[m-1][j-1]);
	for (j = 1; j <= quantization-1; j++)
	  std::swap (Q[j-1][i], Q[j-1][m-1]);
	  Q[j-1][i-1] = Q[j-1][i-1];
    }
    if (x)
    {
      for (i = m + 1; i <= quantization-1; i++)
      {
	if (y = Q[i-1][m - 2])
	{
	  y /= x;
	  Q[i-1][m - 2] = y;
	  for (j = m; j <= quantization-1; j++)
	    Q[i-1][j-1] -= y * Q[m-1][j-1];
	  for (j = 1; j <= quantization-1; j++)
	    Q[j-1][m-1] += y * Q[j-1][i-1];
	}
      }
    }
  }
 
}

void CoOccurrenceExtractor::balanceMatrix()
{

  int last, j, i;
  float s, r, g, f, c, sqrdx;

  sqrdx = RADIX * RADIX;
  last = 0;
  while (last == 0)
  {
    last = 1;
    for (i = 1; i <= quantization-1; i++)
    {
      r = c = 0.0;
      for (j = 1; j <= quantization-1; j++)
			if (j != i)
			{
			  c += fabs (Q[j-1][i-1]);
			  r += fabs (Q[i-1][j-1]);
			}
			
      if (c && r)
      {
		g = r / RADIX;
		f = 1.0;
		s = c + r;
		while (c < g)
		{
		  f *= RADIX;
		  c *= sqrdx;
		}
		g = r * RADIX;
		while (c > g)
		{
		  f /= RADIX;
		  c /= sqrdx;
		}
		if ((c + r) / f < 0.95 * s)
		{
		  last = 0;
		  g = 1.0 / f;
		  for (j = 1; j <= quantization-1; j++)
			Q[i-1][j-1] *= g;
		  for (j = 1; j <= quantization-1; j++)
			Q[j-1][i-1] *= f;
		}
      }
      
    }
  }
  
}

Mat CoOccurrenceExtractor::getImage() const
{
   //IplImage* tmp = cvCreateImage( cvSize( image->width,
                                          //image->height ),
                                  //IPL_DEPTH_8U, 1 );
   //cvCopy(image, tmp );

   //return tmp;
	return image.clone();
}

