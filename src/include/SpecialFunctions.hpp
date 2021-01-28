//
// SpecialFunctions.hpp
//
// Copyright (c) 2020 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//
#include<iostream>
#include<Eigen/Dense>
#include<Eigen/Core>
#include <unsupported/Eigen/SpecialFunctions>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include"utils.hpp"


double normalLogPdf(const Eigen::VectorXd &y, const Eigen::VectorXd &mu, double sigma2);

double normalLogPdf(const Eigen::VectorXd &y, const Eigen::VectorXd &mu, const Eigen::MatrixXd &covMatrix);

double normalPdf(const Eigen::VectorXd &y, const Eigen::VectorXd &mu, double sigma2);

double normalPdf(const Eigen::VectorXd &y, const Eigen::VectorXd &mu, const Eigen::MatrixXd &covMatrix);
