#include <fstream>
#include <vector>
#include "Load_CSV.h"

using namespace Eigen;

Eigen::MatrixXd load_csv(const std::string & path) {

	std::ifstream indata;
	indata.open(path);
	std::string line;
	std::vector<double> values;
	int rows = 0;
	while (getline(indata, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		while (std::getline(lineStream, cell, ',')) {
			values.push_back(std::stod(cell));
		}
		++rows;
	}
	return Eigen::Map<const Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>>(values.data(), rows, values.size() / rows);
}