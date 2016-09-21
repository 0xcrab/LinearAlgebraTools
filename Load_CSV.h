#pragma once

/* This header provides interface that 
	load matrix from csv file to Eigen Matrix
	*/

#include <Eigen/Dense>

Eigen::MatrixXd load_csv(const std::string & path);
