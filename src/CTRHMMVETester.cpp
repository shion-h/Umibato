//
// CTRHMMVETester.cpp
//
// Copyright (c) 2020 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//
#include"include/CTRHMMVETester.hpp"

using namespace std;
using namespace Eigen;
using namespace boost::math;

CTRHMMVETester::CTRHMMVETester(const MatrixXd &x, const MatrixXd &y, const MatrixXd &yVariance, const vector<double> &timepoints, const vector<string> &subjects, int K, double thresholdOfClusterShrinkage, double thresholdOfConvergence, int iterNum)//{{{
    :
     CTRHMMVariationalEstimator(x, y, yVariance, timepoints, subjects, K, thresholdOfClusterShrinkage, thresholdOfConvergence, iterNum)
{
}//}}}

CTRHMMVETester::~CTRHMMVETester(){//{{{

}//}}}

void CTRHMMVETester::testParseMetadata(){//{{{
    this->parseMetadata(_timepoints, _subjects);
    assert(_initPointIdx[0] == 0);
    assert(_endPointIdx[0] == 2);
    assert(_initPointIdx[1] == 3);
    assert(_endPointIdx[1] == 4);
    assert(_S == 2);
    assert(_delta[0] == 1.4);
    assert(_delta[1] == 1.0);
    assert(_deltaIdx[0] == 0);
    assert(_deltaIdx[1] == 1);
    assert(_deltaIdx[2] == -1);
    assert(_deltaIdx[3] == 1);
    assert(_deltaIdx[4] == -1);
    cout<<"parseTrainMetadata(_timepoints, _subjects) test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testInitializeParameters(){//{{{
    this->initializeParameters();
    cout<<"initializeParameters() finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testUpdateForwardProbs(){//{{{
    parseCSV2Eigen("./testcase/input/likelihood_given_z.csv", _logLikelihoodGivenZ);
    _logLikelihoodGivenZ = _logLikelihoodGivenZ.array().log();
    parseCSV2Eigen("./testcase/input/pi.csv", _pi);
    parseCSV2Eigen("./testcase/input/p_delta_t.csv", _PDeltaT[0]);
    parseCSV2Eigen("./testcase/input/p_delta_t.csv", _PDeltaT[1]);
    this->updateForwardProbs();
    MatrixXd trueAlpha;
    parseCSV2Eigen("./testcase/output/alpha.csv", trueAlpha);
    VectorXd trueC;
    parseCSV2Eigen("./testcase/output/c.csv", trueC);
    assert((_logAlpha - trueAlpha.array().log().matrix()).array().abs().sum() < 0.01);
    assert((_c - trueC.array().log().matrix()).array().abs().sum() < 0.01);
    cout<<"updateForwardProbs() test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testUpdateBackwardProbs(){//{{{
    parseCSV2Eigen("./testcase/input/likelihood_given_z.csv", _logLikelihoodGivenZ);
    _logLikelihoodGivenZ = _logLikelihoodGivenZ.array().log();
    parseCSV2Eigen("./testcase/input/p_delta_t.csv", _PDeltaT[0]);
    parseCSV2Eigen("./testcase/input/p_delta_t.csv", _PDeltaT[1]);
    parseCSV2Eigen("./testcase/input/c.csv", _c);
    _c = _c.array().log();
    this->updateBackwardProbs();
    MatrixXd trueBeta;
    parseCSV2Eigen("./testcase/output/beta.csv", trueBeta);
    assert((_logBeta - trueBeta.array().log().matrix()).array().abs().sum() < 0.01);
    cout<<"updateBackwardProbs() test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testUpdateXi(){//{{{
    parseCSV2Eigen("./testcase/input/alpha.csv", _logAlpha);
    _logAlpha = _logAlpha.array().log();
    parseCSV2Eigen("./testcase/input/beta.csv", _logBeta);
    _logBeta = _logBeta.array().log();
    parseCSV2Eigen("./testcase/input/c.csv", _c);
    _c = _c.array().log();
    parseCSV2Eigen("./testcase/input/likelihood_given_z.csv", _logLikelihoodGivenZ);
    _logLikelihoodGivenZ = _logLikelihoodGivenZ.array().log();
    parseCSV2Eigen("./testcase/input/p_delta_t.csv", _PDeltaT[0]);
    parseCSV2Eigen("./testcase/input/p_delta_t.csv", _PDeltaT[1]);
    this->updateXi();
    MatrixXd trueXi0, trueXi1, trueXi3;
    parseCSV2Eigen("./testcase/output/xi0.csv", trueXi0);
    parseCSV2Eigen("./testcase/output/xi1.csv", trueXi1);
    parseCSV2Eigen("./testcase/output/xi3.csv", trueXi3);
    assert((_xi[0] - trueXi0).array().abs().sum() < 0.01);
    assert((_xi[1] - trueXi1).array().abs().sum() < 0.01);
    assert((_xi[3] - trueXi3).array().abs().sum() < 0.01);
    cout<<"updateXi() test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testUpdateC(){//{{{
    parseCSV2Eigen("./testcase/input/xi0.csv", _xi[0]);
    parseCSV2Eigen("./testcase/input/xi1.csv", _xi[1]);
    parseCSV2Eigen("./testcase/input/xi3.csv", _xi[3]);
    this->updateC();
    MatrixXd trueC0, trueC1;
    parseCSV2Eigen("./testcase/output/C0.csv", trueC0);
    parseCSV2Eigen("./testcase/output/C1.csv", trueC1);
    assert((_C[0] - trueC0).array().abs().sum() < 0.01);
    assert((_C[1] - trueC1).array().abs().sum() < 0.01);
    cout<<"updateC() test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testUpdateTauAndN(){//{{{
    parseCSV2Eigen("./testcase/input/q.csv", _Q);
    parseCSV2Eigen("./testcase/input/C0.csv", _C[0]);
    parseCSV2Eigen("./testcase/input/C1.csv", _C[1]);
    parseCSV2Eigen("./testcase/input/p_delta_t.csv", _PDeltaT[0]);
    parseCSV2Eigen("./testcase/input/p_delta_t.csv", _PDeltaT[1]);
    this->updateTauAndN();
    MatrixXd trueTauN;
    parseCSV2Eigen("./testcase/output/ex_tau_n.csv", trueTauN);
    assert((_exTauN - trueTauN).array().abs().sum() < 0.01);
    cout<<"updateTauAndN() test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testUpdateQ(){//{{{
    parseCSV2Eigen("./testcase/input/ex_tau_n.csv", _exTauN);
    this->updateQ();
    MatrixXd trueQ, trueP0, trueP1;
    parseCSV2Eigen("./testcase/output/q.csv", trueQ);
    parseCSV2Eigen("./testcase/output/p_delta_t0.csv", trueP0);
    parseCSV2Eigen("./testcase/output/p_delta_t1.csv", trueP1);
    assert((_Q - trueQ).array().abs().sum() < 0.01);
    assert((_PDeltaT[0] - trueP0).array().abs().sum() < 0.01);
    assert((_PDeltaT[1] - trueP1).array().abs().sum() < 0.01);
    cout<<"updateQ() test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testUpdatePhi(){//{{{
    parseCSV2Eigen("./testcase/input/gamma.csv", _gamma);
    parseCSV2Eigen("./testcase/input/lambda.csv", _lambda);
    this->updatePhi();
    MatrixXd trueExPhi0, trueExPhi1;
    MatrixXd trueQPE;
    parseCSV2Eigen("./testcase/output/ex_phi0.csv", trueExPhi0);
    parseCSV2Eigen("./testcase/output/ex_phi1.csv", trueExPhi1);
    parseCSV2Eigen("./testcase/output/q_phi_entropy.csv", trueQPE);
    assert((_exPhi[0] - trueExPhi0).array().abs().sum() < 0.01);
    assert((_exPhi[1] - trueExPhi1).array().abs().sum() < 0.01);
    assert(abs(_qPhiEntropy - trueQPE(0, 0)) < 0.01);
    for(int k=0; k<_K; k++){
        for(int m=0; m<_y.rows(); m++){
            MatrixXd trueEPPT;
            parseCSV2Eigen("./testcase/output/ex_phi_phi_t" + to_string(k) + to_string(m) + ".csv", trueEPPT);
            assert((_exPhiPhiT[k][m] - trueEPPT).array().abs().sum() < 0.01);
        }
    }
    cout<<"updatePhi() test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testUpdateLambda(){//{{{
    for(int k=0; k<_K; k++){
        for(int m=0; m<_y.rows(); m++){
            parseCSV2Eigen("./testcase/input/ex_phi_phi_t" + to_string(k) + to_string(m) + ".csv", _exPhiPhiT[k][m]);
        }
    }
    this->updateLambda();
    MatrixXd trueLambda;
    parseCSV2Eigen("./testcase/output/lambda.csv", trueLambda);
    assert((_lambda - trueLambda).array().abs().sum() < 0.01);
    cout<<"updateLambda() test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testUpdateLogLikelihoodGivenZ(){//{{{
    parseCSV2Eigen("./testcase/input/ex_phi0.csv", _exPhi[0]);
    parseCSV2Eigen("./testcase/input/ex_phi1.csv", _exPhi[1]);
    for(int k=0; k<_K; k++){
        for(int m=0; m<_y.rows(); m++){
            parseCSV2Eigen("./testcase/input/ex_phi_phi_t" + to_string(k) + to_string(m) + ".csv", _exPhiPhiT[k][m]);
        }
    }
    this->updateLogLikelihoodGivenZ(_x, _y, _yVariance);
    MatrixXd trueLLGZ;
    parseCSV2Eigen("./testcase/output/ex_log_likelihood_given_z.csv", trueLLGZ);
    assert((_logLikelihoodGivenZ - trueLLGZ).array().abs().sum() < 0.01);
    cout<<"updateLikelihoodGivenZ() test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testCalculateELBO(){//{{{
    parseCSV2Eigen("./testcase/input/c.csv", _c);
    _c = _c.array().log();
    parseCSV2Eigen("./testcase/input/lambda.csv", _lambda);
    for(int k=0; k<_K; k++){
        for(int m=0; m<_y.rows(); m++){
            parseCSV2Eigen("./testcase/input/ex_phi_phi_t" + to_string(k) + to_string(m) + ".csv", _exPhiPhiT[k][m]);
        }
    }
    _qPhiEntropy = 0;
    this->calculateELBO();
    VectorXd trueELBO;
    parseCSV2Eigen("./testcase/output/ELBO.csv", trueELBO);
    assert(abs(_ELBO[_ELBO.size()-1] - trueELBO(0)) < 0.01);
    cout<<"calculateELBO() test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testFindViterbiPath(){//{{{
    parseCSV2Eigen("./testcase/input/likelihood_given_z.csv", _logLikelihoodGivenZ);
    _logLikelihoodGivenZ = _logLikelihoodGivenZ.array().log();
    parseCSV2Eigen("./testcase/input/p_delta_t.csv", _PDeltaT[0]);
    parseCSV2Eigen("./testcase/input/p_delta_t.csv", _PDeltaT[1]);
    parseCSV2Eigen("./testcase/input/pi.csv", _pi);
    this->findViterbiPath();
    MatrixXd trueMaxPathLogProb;
    MatrixXi trueMaxPath;
    VectorXi trueViterbiPath;
    parseCSV2Eigen("./testcase/output/max_path_log_prob.csv", trueMaxPathLogProb);
    parseCSV2Eigen("./testcase/output/max_path.csv", trueMaxPath);
    parseCSV2Eigen("./testcase/output/viterbi_path.csv", trueViterbiPath);
    assert((_maxPathLogProb - trueMaxPathLogProb).array().abs().sum() < 0.01);
    assert((_maxPath - trueMaxPath).array().abs().sum() < 0.01);
    assert((_ViterbiPath - trueViterbiPath).array().abs().sum() < 0.01);
    cout<<"testFindViterbiPath() test finished successfully."<<endl;
}//}}}

void CTRHMMVETester::testAll(){//{{{
    this->testParseMetadata();
    this->testInitializeParameters();
    this->testUpdateForwardProbs();
    this->testUpdateBackwardProbs();
    this->testUpdateXi();
    this->testUpdateC();
    this->testUpdateTauAndN();
    this->testUpdateQ();
    this->testUpdatePhi();
    this->testUpdateLambda();
    this->testUpdateLogLikelihoodGivenZ();
    this->testCalculateELBO();
    this->testFindViterbiPath();
}//}}}
