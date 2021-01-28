//
// SpecialFunctions.cpp
//
// Copyright (c) 2020 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//
#include"include/SpecialFunctions.hpp"

using namespace std;
using namespace Eigen;

double normalLogPdf(const VectorXd &y, const VectorXd &mu, double sigma2){
    MatrixXd covMatrix(sigma2 * MatrixXd::Identity(y.size(), y.size()));
    return normalLogPdf(y, mu, covMatrix);
}

double normalLogPdf(const VectorXd &y, const VectorXd &mu, const MatrixXd &covMatrix){
    double term1 = (y.size() / 2) * log(2 * M_PI) + (1 / 2) * logdet(covMatrix);
    double term2 = (y - mu).transpose() * covMatrix.inverse() * (y - mu);
    return - term1 - term2 / 2;
}

double normalPdf(const VectorXd &y, const VectorXd &mu, double sigma2){
    MatrixXd covMatrix(sigma2 * MatrixXd::Identity(y.size(), y.size()));
    return normalPdf(y, mu, covMatrix);
}

double normalPdf(const VectorXd &y, const VectorXd &mu, const MatrixXd &covMatrix){
    double term1 = sqrt(pow((2 * M_PI), y.size() * covMatrix.determinant()));
    double term2 = (y - mu).transpose() * covMatrix.inverse() * (y - mu);
    return (1 / term1) * exp(- term2 / 2);
}
