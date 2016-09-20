#pragma once

/* This file includes linear algebra tools given in MTH9821 Textbook
	Author: Zilun Shen
	Email: shenzilun@gmail.com
	Date: 08/30/2016

	- Forward substitution
	- Backward substitution
	- LU decomposition without row pivoting
	- LU decomposition with row pivoting
	*/

#include <Eigen/Dense>
#include <tuple>
#include <iostream>

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;


// Forward Substitution
Vector forward_subst(const Matrix& L, const Vector& b);

// Backward Substitution
Vector backward_subst(const Matrix& U, const Vector& b);

// Forward substitution for lower triangular bidiagonal matrix
Vector forward_subst_bidiag(const Matrix& L, const Vector& b);

// Backward Substitution for upper tiangular bidiagonal matrix
Vector backward_subst_bidiag(const Matrix& U, const Vector& b);

// Forward substitution for lower triangular banded matrix
Vector forward_subst_banded(const Matrix& L, const Vector& b, int m);

// Backward Substitution for upper tiangular banded matrix
Vector backward_subst_banded(const Matrix& U, const Vector& b, int m);

/* LU decomposition without pivoting
	A = LU
	*/
std::tuple<Matrix, Matrix> lu_no_pivoting(Matrix A);

/* LU decomposition with pivoting
	A = LU
	*/
std::tuple<Matrix, Matrix, Matrix> lu_row_pivoting(Matrix A);

/* LU decomposition with pivoting for banded matrix
A = LU
*/
std::tuple<Matrix, Matrix, Matrix> lu_row_pivoting_banded(Matrix A, int m);

/* Cholesky decomposition
	A = UtU
	*/
Matrix cholesky(Matrix A);

/* Cholesky decomposition for m-banded spd matrix
	A(i,j)=0 for abs(i-j)>m
	*/
Matrix cholesky_banded(Matrix A, int m);

/* Cholesky decomposition for tridiagonal spd matrix
	i.e. 1-banded spd matrix*/
Matrix cholesky_tridiag_spd(Matrix A);

/* Linear solver using Cholesky decomposition for
	spd matrix
	*/
Vector linear_solve_cholesky(const Matrix& A, const Vector& b);

/* Linear solver using Cholesky decomposition for
	tridiagonal spd matrix
	*/
Vector linear_solve_cholesky_tridiag(const Matrix& A, const Vector& b);

/* Linear solver using Cholesky decomposition for
	banded spd matrix
	*/
Vector linear_solve_cholesky_banded(const Matrix& A, const Vector& b, int m);

/* Linear solver using LU decomposition with ro 
	pivoting
	*/

Vector linear_solve_lu_row_pivot(const Matrix& A, const Vector& b);

/* Use LU decomposition with row pivoting to find matrix inverse
	The return values are:
	Ainv, Linv, Uinv, P
*/
std::tuple<Matrix, Matrix, Matrix, Matrix>  inverse(const Matrix& A);


/* Define Stop Criterion
	- Residual-based stopping criterion
	- Consecutive approximation stopping criterion*/
enum class Criterion_Type { Resiudal_Based, Consec_Approx };

class StopCriterion {
public:
	StopCriterion(double _tol, Criterion_Type _type, const Vector& _r0) 
		: tol(_tol), stop_iter_residual(_tol*_r0.norm()), type(_type) {}
	
	bool operator()(const Vector& x_old, const Vector& x_new, const Vector& r) {

		if (type == Criterion_Type::Resiudal_Based)
			return r.norm() > stop_iter_residual;
		else
			return (x_old - x_new).norm() > tol;
	}

private:
	double tol;
	double stop_iter_residual;
	Criterion_Type type;
};

/* Jacobi Iteration*/
std::tuple<Vector, int> linear_solve_jacobi_iter(const Matrix& A, const Vector& b, 
	const Vector& x0, double tol, Criterion_Type type);

/* Gauss-Siedel Iteration*/
std::tuple<Vector, int> linear_solve_gs_iter(const Matrix& A, const Vector& b,
	const Vector& x0, double tol, Criterion_Type type);

/* SOR Iteration*/
std::tuple<Vector, int> linear_solve_sor_iter(const Matrix& A, const Vector& b,
	const Vector& x0, double tol, double w, Criterion_Type type);