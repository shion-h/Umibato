//
// CTRHMMVariationalEstimator.hpp
//
// Copyright (c) 2020 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//

#ifndef CTRHMMVE
#define CTRHMMVE

#include<stdlib.h>
#include<math.h>
#include<cmath>
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
#include<Eigen/LU>
#include<unsupported/Eigen/SpecialFunctions>
#include<unsupported/Eigen/MatrixFunctions>
#include"SpecialFunctions.hpp"
#include"utils.hpp"


extern std::random_device rd;
extern unsigned int rdValue;
extern std::mt19937 gen;

void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove);
void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove);

class CTRHMMVariationalEstimator{
protected:
    const Eigen::MatrixXd &_x, &_y, &_yVariance;
    std::vector<double> _timepoints;
    std::vector<std::string> _subjects;
    std::vector<int> _initPointIdx, _endPointIdx, _deltaIdx;
    std::vector<double> _delta;
    const int _N;
    int _K, _S;
    Eigen::MatrixXd _logLikelihoodGivenZ;
    Eigen::MatrixXd _Q;
    Eigen::MatrixXd _exTauN;
    Eigen::VectorXd _c;
    Eigen::MatrixXd _logAlpha, _logBeta, _gamma;
    Eigen::VectorXd _pi;
    Eigen::MatrixXi _maxPath;
    Eigen::VectorXi _ViterbiPath;
    Eigen::MatrixXd _maxPathLogProb;
    std::vector<Eigen::MatrixXd> _xi, _PDeltaT, _C;
    Eigen::VectorXd _lambda;
    std::vector<Eigen::MatrixXd> _exPhi;
    std::vector<std::vector<Eigen::MatrixXd> > _exPhiPhiT;
    std::vector<double> _ELBO;
    double _qPhiEntropy;
    double _residualSum, _eta;
    const double _thresholdOfClusterShrinkage, _thresholdOfConvergence;
    const int _iterNum;
public:
    CTRHMMVariationalEstimator(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, const Eigen::MatrixXd &yVariance, const std::vector<double> &timepoints, const std::vector<std::string> &subjects, int K, double thresholdOfClusterShrinkage, double thresholdOfConvergence, int iterNum);
    virtual ~CTRHMMVariationalEstimator();
    virtual void parseMetadata(const std::vector<double> &timepoints, const std::vector<std::string> &subjects);
    virtual void initializeParameters();
    virtual void updateExpectations();
    virtual void updateForwardProbs();
    virtual void updateBackwardProbs();
    virtual void updateGamma();
    virtual void updateXi();
    virtual void updateC();
    virtual void updateTauAndN();
    virtual void updateQ();
    virtual void updatePDeltaT();
    virtual void updatePhi();
    virtual void updateLambda();
    virtual void updateLogLikelihoodGivenZ(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, const Eigen::MatrixXd &yVariance);
    virtual void updatePi();
    virtual void updateEta();
    virtual void calcAndWriteTestLogLikelihood(const Eigen::MatrixXd &testX, const Eigen::MatrixXd &testY, const Eigen::MatrixXd &testYVariance, const std::vector<double> &testTimepoints, const std::vector<std::string> &testSubjects, const std::string outputDirectory);
    virtual void calculateELBO();
    virtual void findViterbiPath();
    virtual void writeParameters(std::string outputDirectory)const;
    virtual void writeObjective(std::string outputDirectory)const;
    virtual bool isConvergence()const;
    virtual void shrinkCluster(int k);
    virtual bool checkNonRelevantCluster();
    virtual void runIteraions();
};

#endif
