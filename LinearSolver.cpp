/* This file includes linear algebra tools given in MTH9821 Textbook
Author: Zilun Shen
Email: shenzilun@gmail.com
Date: 08/30/2016

- Forward substitution
- Backward substitution
- LU decomposition without row pivoting
- LU decomposition with row pivoting
*/

#include "LinearSolver.h"
#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <numeric>
#include <stdexcept>

/* Define Matrix and Vector class
These classes are derived from MatrixXd and VectorXd
They support index beginning from 1 to n,
while in Eigen index begin with 0
*/

// Forward Substitution
Vector forward_subst(const Matrix& L, const Vector& b) {

	assert(L.rows() == L.cols());
	assert(L.rows() == b.rows());
	Vector x(b.rows());
	double sum;
	int n = L.cols();
	x(0) = b(0) / L(0, 0);
	for (int j = 1; j < n; j++) {
		sum = 0;
		for (int k = 0; k < j; k++) {
			sum += L(j, k)*x(k);
		}
		x(j) = (b(j) - sum) / L(j, j);
	}
	return x;
}

// Backward Substitution
Vector backward_subst(const Matrix& U, const Vector& b) {

	assert(U.rows() == U.cols());
	assert(U.rows() == b.rows());
	Vector x(b.rows());
	double sum;
	int n = U.cols();
	x(n - 1) = b(n - 1) / U(n - 1, n - 1);
	for (int j = n - 2; j >= 0; j--) {
		sum = 0;
		for (int k = j + 1; k < n; k++) {
			sum += U(j, k)*x(k);
		}
		x(j) = (b(j) - sum) / U(j, j);
	}
	return x;
}

// Forward substitution for lower triangular bidiagonal matrix
Vector forward_subst_bidiag(const Matrix& L, const Vector& b) {

	assert(L.rows() == L.cols());
	assert(L.rows() == b.rows());
	const int n = b.rows();
	Vector x(n);
	x(0) = b(0) / L(0, 0);
	for (int j = 1; j < n; j++) {
		x(j) = (b(j) - L(j, j - 1)*x(j - 1)) / L(j, j);
	}
	return x;
}

// Backward Substitution for upper triangular bidiagonal matrix
Vector backward_subst_bidiag(const Matrix& U, const Vector& b) {

	assert(U.rows() == U.cols());
	assert(U.rows() == b.rows());
	Vector x(b.rows());
	int n = U.cols();
	x(n - 1) = b(n - 1) / U(n - 1, n - 1);
	for (int j = n - 2; j >= 0; j--) {
		x(j) = (b(j) - U(j, j + 1)*x(j + 1)) / U(j, j);
	}
	return x;
}

// Forward substitution for lower triangular banded matrix
Vector forward_subst_banded(const Matrix& L, const Vector& b, int m) {

	assert(L.rows() == L.cols());
	assert(L.rows() == b.rows());
	Vector x(b.rows());
	double sum;
	int n = L.cols();
	x(0) = b(0) / L(0, 0);
	for (int j = 1; j < n; j++) {
		sum = 0;
		int ub = std::min(j, j + m + 1);
		for (int k = 0; k < ub; k++) {
			sum += L(j, k)*x(k);
		}
		x(j) = (b(j) - sum) / L(j, j);
	}
	return x;
}

// Backward Substitution for upper tiangular banded matrix
Vector backward_subst_banded(const Matrix& U, const Vector& b, int m) {

	assert(U.rows() == U.cols());
	assert(U.rows() == b.rows());
	Vector x(b.rows());
	double sum;
	int n = U.cols();
	x(n - 1) = b(n - 1) / U(n - 1, n - 1);
	for (int j = n - 2; j >= 0; j--) {
		sum = 0;
		int ub = std::min(n, j + m + 1);
		for (int k = j + 1; k < ub; k++) {
			sum += U(j, k)*x(k);
		}
		x(j) = (b(j) - sum) / U(j, j);
	}
	return x;
}

void lu_helper(Matrix &A, Matrix &L, Matrix &U, int i, int n) {
	if (A(i, i) == 0) {
		throw std::overflow_error("Devided By 0 Error");
	}
	for (int k = i; k < n; k++) {
		U(i, k) = A(i, k);
		L(k, i) = A(k, i) / U(i, i);
	}
	for (int j = i + 1; j < n; j++) {
		for (int k = i + 1; k < n; k++) {
			A(j, k) -= L(j, i)*U(i, k);
		}
	}
}

/* LU decomposition without pivoting
A = LU
*/
std::tuple<Matrix, Matrix> lu_no_pivoting(Matrix A) {

	assert(A.cols() == A.rows());
	const int n = A.cols();
	Matrix L = Matrix::Zero(n, n);
	Matrix U = Matrix::Zero(n, n);
	for (int i = 0; i < n - 1; i++) {
		lu_helper(A, L, U, i, n);
	}
	L(n - 1, n - 1) = 1;
	U(n - 1, n - 1) = A(n - 1, n - 1);
	return std::make_tuple(std::move(L), std::move(U));
}

/* LU decomposition with pivoting
A = LU
*/
std::tuple<Matrix, Matrix, Matrix> lu_row_pivoting(Matrix A) {

	assert(A.cols() == A.rows());
	const int n = A.cols();
	Matrix L = Matrix::Zero(n, n);
	Matrix U = Matrix::Zero(n, n);
	Matrix P = Matrix::Identity(n, n);

	// Following part is given by psudocode
	for (int i = 0; i < n - 1; i++) {

		int i_max_in_block;
		A.col(i).bottomRows(n - i).cwiseAbs().maxCoeff(&i_max_in_block);
		int i_max = i + i_max_in_block;

		// Switch rows i and i_max of A and P
		A.row(i).swap(A.row(i_max));
		P.row(i).swap(P.row(i_max));
		L.row(i).swap(L.row(i_max));
		lu_helper(A, L, U, i, n);
	}
	L(n - 1, n - 1) = 1;
	U(n - 1, n - 1) = A(n - 1, n - 1);
	return std::make_tuple(std::move(P), std::move(L), std::move(U));
}

/* LU decomposition with pivoting for banded matrix
A = LU
*/
std::tuple<Matrix, Matrix, Matrix> lu_row_pivoting_banded(Matrix A, int m) {

	assert(A.cols() == A.rows());
	const int n = A.cols();
	Matrix L = Matrix::Zero(n, n);
	Matrix U = Matrix::Zero(n, n);
	Matrix P = Matrix::Identity(n, n);

	// Following part is given by psudocode
	for (int i = 0; i < n - 1; i++) {

		int i_max_in_block;
		A.col(i).bottomRows(n - i).cwiseAbs().maxCoeff(&i_max_in_block);
		int i_max = i + i_max_in_block;

		// Switch rows i and i_max of A and P
		A.row(i).swap(A.row(i_max));
		P.row(i).swap(P.row(i_max));
		L.row(i).swap(L.row(i_max));
		if (A(i, i) == 0) {
			throw std::overflow_error("Devided By 0 Error");
		}
		int ub = std::min(n, i + m + 1);
		for (int k = i; k < ub; k++) {
			U(i, k) = A(i, k);
			L(k, i) = A(k, i) / U(i, i);
		}
		for (int j = i + 1; j < n; j++) {
			int ub = std::min(n, j + m + 1);
			for (int k = i + 1; k < ub; k++) {
				A(j, k) -= L(j, i)*U(i, k);
			}
		}
	}
	L(n - 1, n - 1) = 1;
	U(n - 1, n - 1) = A(n - 1, n - 1);
	return std::make_tuple(std::move(P), std::move(L), std::move(U));
}

Matrix cholesky(Matrix A) {

	assert(A.cols() == A.rows());
	if (!A.isApprox(A.transpose())) {
		throw std::overflow_error("The matrix is not symmetric");
	}
	const int n = A.cols();
	Matrix U = Matrix::Zero(n, n);
	for (int i = 0; i < n - 1; i++) {
		if (A(i, i) <= 0) {
			throw std::overflow_error("The matrix is not SPD");
		}
		U(i, i) = sqrt(A(i, i));
		for (int k = i + 1; k < n; k++) {
			U(i, k) = A(i, k) / U(i, i);
		}
		for (int j = i + 1; j < n; j++)
			for (int k = j; k < n; k++) {
				A(j, k) -= U(i, j)*U(i, k);
			}
	}
	if (A(n - 1, n - 1) <= 0) {
		throw std::overflow_error("The matrix is not SPD");
	}
	U(n - 1, n - 1) = sqrt(A(n - 1, n - 1));
	return U;
}

Matrix cholesky_banded(Matrix A, int m) {

	assert(A.cols() == A.rows());
	if (!A.isApprox(A.transpose())) {
		throw std::overflow_error("The matrix is not symmetric");
	}
	const int n = A.cols();
	Matrix U = Matrix::Zero(n, n);
	for (int i = 0; i < n - 1; i++) {
		if (A(i, i) <= 0) {
			throw std::overflow_error("The matrix is not SPD");
		}
		U(i, i) = sqrt(A(i, i));
		int ub = std::min(n, i + m + 1);
		for (int k = i + 1; k < ub; k++) {
			U(i, k) = A(i, k) / U(i, i);
		}
		ub = std::min(n, i + m + 2);
		for (int j = i + 1; j < ub; j++)
			for (int k = j; k < ub; k++) {
				A(j, k) -= U(i, j)*U(i, k);
			}
	}
	if (A(n - 1, n - 1) <= 0) {
		throw std::overflow_error("The matrix is not SPD");
	}
	U(n - 1, n - 1) = sqrt(A(n - 1, n - 1));
	return U;
}

Matrix cholesky_tridiag_spd(Matrix A) {

	assert(A.cols() == A.rows());
	if (!A.isApprox(A.transpose())) {
		throw std::overflow_error("The matrix is not symmetric");
	}
	const int n = A.cols();
	Matrix U = Matrix::Zero(n, n);
	for (int i = 0; i < n - 1; i++) {
		if (A(i, i) <= 0) {
			throw std::overflow_error("The matrix is not SPD");
		}
		U(i, i) = sqrt(A(i, i));
		U(i, i + 1) = A(i, i + 1) / U(i, i);
		A(i + 1, i + 1) -= U(i, i + 1) * U(i, i + 1);
	}
	if (A(n - 1, n - 1) <= 0) {
		throw std::overflow_error("The matrix is not SPD");
	}
	U(n - 1, n - 1) = sqrt(A(n - 1, n - 1));
	return U;
}

Vector linear_solve_cholesky(const Matrix& A, const Vector& b) {
	assert(A.cols() == A.rows());
	assert(A.rows() == b.rows());
	Matrix U = cholesky(A);
	Vector y = forward_subst(U.transpose(), b);
	Vector x = backward_subst(U, y);
	return x;
}

Vector linear_solve_cholesky_tridiag(const Matrix& A, const Vector& b) {

	assert(A.cols() == A.rows());
	assert(A.rows() == b.rows());
	Matrix U = cholesky_tridiag_spd(A);
	Vector y = forward_subst_bidiag(U.transpose(), b);
	Vector x = backward_subst_bidiag(U, y);
	return x;
}

Vector linear_solve_cholesky_banded(const Matrix & A, const Vector & b, int m) {

	assert(A.cols() == A.rows());
	assert(A.rows() == b.rows());
	Matrix U = cholesky_banded(A, m);
	Vector y = forward_subst_banded(U.transpose(), b, m);
	Vector x = backward_subst_banded(U, y, m);
	return x;
}

std::tuple<Matrix, Matrix, Matrix, Matrix> inverse(const Matrix & A)
{
	assert(A.cols() == A.rows());
	Matrix P, L, U;
	std::tie(P, L, U) = lu_row_pivoting(A);

	const int n = A.cols();
	Matrix Linv = Matrix::Zero(n, n);
	Matrix Uinv = Matrix::Zero(n, n);
	
	for (int i = 0; i < n; i++) {

		Vector e = Vector::Zero(n);
		e[i] = 1;
		Linv.col(i) = forward_subst(L, e);
		Uinv.col(i) = backward_subst(U, e);
	}

	Matrix Ainv = Uinv * Linv * P;
	return std::make_tuple(Ainv, Linv, Uinv, P);
}

Vector linear_solve_lu_row_pivot(const Matrix & A, const Vector & b)
{
	assert(A.cols() == A.rows());
	assert(A.rows() == b.rows());
	Matrix P, L, U;
	std::tie(P, L, U) = lu_row_pivoting(A);
	Vector v = forward_subst(L, P*b);
	v = backward_subst(U, v);
	return v;
}

std::tuple<Vector, int> linear_solve_jacobi_iter(const Matrix & A, const Vector & b, const Vector & x0, 
	double tol, Criterion_Type type)
{
	Vector x_new = x0;
	// Init value of x_old that dissatisfy stop criterion
	Vector x_old = x_new + Vector::Constant(x0.size(), tol);
	Vector r = b - A * x0;
	Matrix Dinv(A.diagonal().asDiagonal().inverse());
	Matrix U = A.triangularView<Eigen::StrictlyUpper>();
	Matrix L = A.triangularView<Eigen::StrictlyLower>();
	Vector b_new = Dinv * b;
	int ic = 0;

	StopCriterion stop_crtr(tol, type, r);

	while (stop_crtr(x_old, x_new, r)) {
		x_old = x_new;
		x_new = -Dinv * (L*x_old + U*x_old) + b_new;
		r = b - A * x_new;
		ic++;
	}
	return std::make_tuple(x_new, ic);
}

std::tuple<Vector, int> linear_solve_gs_iter(const Matrix & A, const Vector & b, const Vector & x0,
	double tol, Criterion_Type type)
{
	Vector x_new = x0;
	// Init value of x_old that dissatisfy stop criterion
	Vector x_old = x_new + Vector::Constant(x0.size(), tol);
	Vector r = b - A * x0;
	Matrix D(A.diagonal().asDiagonal());
	Matrix Dinv(D.inverse());
	Matrix U = A.triangularView<Eigen::StrictlyUpper>();
	Matrix L = A.triangularView<Eigen::StrictlyLower>();
	Vector b_new = forward_subst(D+L, b);
	int ic = 0;

	StopCriterion stop_crtr(tol, type, r);

	while (stop_crtr(x_old, x_new, r)) {
		x_old = x_new;
		x_new = -forward_subst(D + L, U*x_old) + b_new;
		r = b - A * x_new;
		ic++;
	}
	return std::make_tuple(x_new, ic);
}

std::tuple<Vector, int> linear_solve_sor_iter(const Matrix & A, const Vector & b, const Vector & x0, 
	double w, double tol, Criterion_Type type)
{
	assert(w > 0 && w < 2);
	Vector x_new = x0;
	// Init value of x_old that dissatisfy stop criterion
	Vector x_old = x_new + Vector::Constant(x0.size(), tol);
	Vector r = b - A * x0;
	Matrix D(A.diagonal().asDiagonal());
	Matrix U = A.triangularView<Eigen::StrictlyUpper>();
	Matrix L = A.triangularView<Eigen::StrictlyLower>();
	Vector b_new = w * forward_subst(D + w*L, b);
	int ic = 0;

	StopCriterion stop_crtr(tol, type, r);

	while (stop_crtr(x_old, x_new, r)) {
		x_old = x_new;
		x_new = forward_subst(D + w*L, (1-w)*D*x_old-w*U*x_old) + b_new;
		r = b - A * x_new;
		ic++;
	}
	return std::make_tuple(x_new, ic);
}
