//
// CTRHMMVETester.hpp
//
// Copyright (c) 2020 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//

#include<stdlib.h>
#include<math.h>
#include<cmath>
#include<cassert>
#include<iostream>
#include<vector>
#include<numeric>
#include<memory>
#include<random>
#include<iomanip>
#include<fstream>
#include<limits>
#include<algorithm>
#include<Eigen/Dense>
#include<Eigen/Core>
#include<unsupported/Eigen/SpecialFunctions>
#include<unsupported/Eigen/MatrixFunctions>
#include"SpecialFunctions.hpp"
#include"utils.hpp"
#include"CTRHMMVariationalEstimator.hpp"


class CTRHMMVETester :public CTRHMMVariationalEstimator{
public:
    CTRHMMVETester(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, const Eigen::MatrixXd &yVariance, const std::vector<double> &timepoints, const std::vector<std::string> &subjects, int K, double thresholdOfClusterShrinkage, double thresholdOfConvergence, int iterNum);
    virtual ~CTRHMMVETester();
    virtual void testParseMetadata();
    virtual void testInitializeParameters();
    virtual void testUpdateForwardProbs();
    virtual void testUpdateBackwardProbs();
    virtual void testUpdateXi();
    virtual void testUpdateC();
    virtual void testUpdateTauAndN();
    virtual void testUpdateQ();
    virtual void testUpdatePhi();
    virtual void testUpdateLambda();
    virtual void testUpdateLogLikelihoodGivenZ();
    virtual void testCalculateELBO();
    virtual void testFindViterbiPath();
    virtual void testAll();
};
