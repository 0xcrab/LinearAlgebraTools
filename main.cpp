#include <iostream>
#include <array>
#include "LinearSolver.h"
#include <vector>
#include <numeric>
#include <iomanip>

#define PRINTLINE std::cout << std::endl\
					 << "-----------------------"\
					 << std::endl << std::endl;
#define PRINT(v) std::cout<<#v" = "<<std::endl<<v<<std::endl;\
				 PRINTLINE
#define TITLE(title) std::cout << "****************** " << #title << " ******************" << std::endl;

void testDecompositionSolvers() {

	Matrix M = Matrix::Random(4, 4) * 100;
	M << 10, 1, 0, 0,
		1, 9, 2, 0,
		0, 2, 11, 2,
		0, 0, 2, 10;

	// Test LU decomposition w/o row pivoting
	Matrix L(4, 4), U(4, 4);
	try {
		std::tie(L, U) = lu_no_pivoting(M);
		PRINT(L);
		PRINT(U);
		PRINT(L*U);
		PRINT(M);
	}
	catch (std::overflow_error e) {
		std::cout << e.what() << std::endl;
	}

	// Test LU decomposition with row pivoting
	Matrix P(4, 4);
	try {
		std::tie(P, L, U) = lu_row_pivoting(M);
		PRINT(L);
		PRINT(U);
		PRINT(P);
		PRINT(L*U);
		PRINT(P*M);
	}
	catch (std::overflow_error e) {
		std::cout << e.what() << std::endl;
	}

	// Test LU decomposition with row pivoting for banded matrix
	try {
		std::tie(P, L, U) = lu_row_pivoting_banded(M, 1);
		PRINT(L);
		PRINT(U);
		PRINT(P);
		PRINT(L*U);
		PRINT(P*M);
	}
	catch (std::overflow_error e) {
		std::cout << e.what() << std::endl;
	}

	// Test cholesky
	try {
		Matrix U = cholesky(M);
		PRINT(U);
		PRINT(U.transpose()*U);
		PRINT(M);
	}
	catch (std::overflow_error e) {
		std::cout << e.what() << std::endl;
	}

	// Test cholesky for 1-banded matrix
	try {
		Matrix U = cholesky_banded(M, 1);
		PRINT(U);
		PRINT(U.transpose()*U);
		PRINT(M);
	}
	catch (std::overflow_error e) {
		std::cout << e.what() << std::endl;
	}

	// Test cholesky for tridiagonal matrix
	try {
		Matrix U = cholesky_tridiag_spd(M);
		PRINT(U);
		PRINT(U.transpose()*U);
		PRINT(M);
	}
	catch (std::overflow_error e) {
		std::cout << e.what() << std::endl;
	}

	// Test cholesky solver for tridiagonal matrix
	try {
		Vector b(4, 1);
		b << 1, 2, 3, 4;
		Vector X = linear_solve_cholesky(M, b);
		PRINT(X);
		PRINT(M*X);
	}
	catch (...) {}

	// Test cholesky solver for tridiagonal matrix
	try {
		Vector b(4, 1);
		b << 1, 2, 3, 4;
		Vector X = linear_solve_cholesky_tridiag(M, b);
		PRINT(X);
		PRINT(M*X);
	}
	catch (...) {}

	// Test cholesky solver for banded matrix
	try {
		Vector b(4, 1);
		b << 1, 2, 3, 4;
		Vector X = linear_solve_cholesky_banded(M, b, 1);
		PRINT(X);
		PRINT(M*X);
	}
	catch (...) {}
}

void testIterationSolvers() {

	int N = 14;
	Matrix A = Matrix::Zero(N, N);
	A.diagonal() = Vector::Constant(N, 2);
	A.diagonal(1) = Vector::Constant(N - 1, -1);
	A.diagonal(-1) = Vector::Constant(N - 1, -1);
	PRINT(A);
	PRINT(A.diagonal());
	Vector b(N);
	for (int i = 0; i < N; i++) b[i] = i*i;
	PRINT(b);
	Vector x;
	int ic;

	std::cout.setf(std::ios::fixed);
	std::cout.precision(9);

	// test Jacobi-iteration
	TITLE(Jacobi Residual_Based);
	std::tie(x, ic) = linear_solve_jacobi_iter(A, b, Vector::Ones(N), 1e-6, Criterion_Type::Resiudal_Based);
	PRINT(x);
	PRINT(ic);
	TITLE(Jacobi Consec_Approx);
	std::tie(x, ic) = linear_solve_jacobi_iter(A, b, Vector::Ones(N), 1e-6, Criterion_Type::Consec_Approx);
	PRINT(x);
	PRINT(ic);

	// test GS-iteration
	TITLE(GS Residual_Based);
	std::tie(x, ic) = linear_solve_gs_iter(A, b, Vector::Ones(N), 1e-6, Criterion_Type::Resiudal_Based);
	PRINT(x);
	PRINT(ic);
	TITLE(GS Consec_Approx);
	std::tie(x, ic) = linear_solve_gs_iter(A, b, Vector::Ones(N), 1e-6, Criterion_Type::Consec_Approx);
	PRINT(x);
	PRINT(ic);

	// test SOR-iteration
	TITLE(SOR Residual_Based);
	std::tie(x, ic) = linear_solve_sor_iter(A, b, Vector::Ones(N), 1.15, 1e-6, Criterion_Type::Resiudal_Based);
	PRINT(x);
	PRINT(ic);
	TITLE(SOR Consec_Approx);
	std::tie(x, ic) = linear_solve_sor_iter(A, b, Vector::Ones(N), 1.15, 1e-6, Criterion_Type::Consec_Approx);
	PRINT(x);
	PRINT(ic);

	// Problem 3.4
	TITLE(PROBLEM 3.4);
	std::vector<double> w;
	for (int k = 102; k <= 198; k += 2)
		w.push_back(k/100.0);
	for (auto cur_w : w) {
		std::tie(std::ignore, ic) = linear_solve_sor_iter(A, b, Vector::Ones(N), cur_w, 1e-6, Criterion_Type::Resiudal_Based);
		std::cout << ic << std::endl;//std::cout << std::setprecision(3) << cur_w << "\t" << ic << std::endl;
	}
}

int main() {

	//testDecompositionSolvers();

	testIterationSolvers();
	return 0;
}