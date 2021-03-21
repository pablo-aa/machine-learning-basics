/*
 * SymmetricMatrix.h
 * This file is part of SIAC (Siatema Inteligente para Análise do Cascalho)
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
Author: Alan Zanoni Peixinho (alan-peixinho@hotmail.com)
*/

#ifndef SYMMETRIC_MATRIX_H
#define SYMMETRIC_MATRIX_H

#include <algorithm>

template<class Type> class SymmetricMatrix
{
	private:
	Type* matrix;
	unsigned length;
	
	public:
	
	typedef Type* iterator;
	typedef const Type* const_iterator;
	
    SymmetricMatrix(const unsigned size)
	{
		length = ((size*(size+1))/2) + size;
		matrix = new Type[length];
	}
	
    SymmetricMatrix(unsigned size, const Type& initialValue)
	{
		length = ((size*(size+1))/2) + size;
		matrix = new Type[length];
		std::fill(begin(), end(), initialValue);
	}
	
    SymmetricMatrix(const SymmetricMatrix &m)
	{
		length = m.end() - m.begin();
		matrix = new Type[length];
		std::copy(m.begin(), m.end(), matrix);
	}
	
    ~SymmetricMatrix()
	{
		delete matrix;
	}
	
	//somente para consulta, nao permite alteracao, utilizado para variaveis const (não devem ser alteradas)
	const_iterator begin() const
	{
		return matrix;
	}
	
	const_iterator end() const
	{
		return matrix+length;
	}
	
	//permite alteracao nos dados
	iterator begin()
	{
		return matrix;
	}
	
	iterator end()
	{
		return matrix+length;
	}
	
	const Type& at(unsigned row, unsigned col) const
	{
		return (row<col)?
			matrix[(row*(row+1))/2 + col]
			:matrix[(col*(col+1))/2 + row];
	}
	
	Type& at(unsigned row, unsigned col)
	{
		return (row<col)?
			matrix[(row*(row+1))/2 + col]
			:matrix[(col*(col+1))/2 + row];
	}
};

#endif//SYMMETRIC_MATRIX_H
