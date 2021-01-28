//
// CTRHMMVariationalEstimator.cpp
//
// Copyright (c) 2020 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//
#include"include/CTRHMMVariationalEstimator.hpp"

using namespace std;
using namespace Eigen;
using namespace boost::math;

random_device rd;
unsigned int rdValue = rd();
// unsigned int rdValue = 1265339876;
mt19937 gen(rdValue);

void removeElement(VectorXd &vector, unsigned int indexToRemove){//{{{
    vector.segment(indexToRemove, vector.size() - 1 - indexToRemove) = vector.tail(vector.size() - 1 - indexToRemove);
    vector.conservativeResize(vector.size()-1, 1);
}//}}}

void removeRow(Eigen::MatrixXd &matrix, unsigned int rowToRemove){//{{{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);

    matrix.conservativeResize(numRows,numCols);
}//}}}

void removeColumn(Eigen::MatrixXd &matrix, unsigned int colToRemove){//{{{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.rightCols(numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}//}}}

CTRHMMVariationalEstimator::CTRHMMVariationalEstimator(const MatrixXd &x, const MatrixXd &y, const MatrixXd &yVariance, const vector<double> &timepoints, const vector<string> &subjects, int K, double thresholdOfClusterShrinkage, double thresholdOfConvergence, int iterNum)//{{{
    :
    _x(x),
    _y(y),
    _yVariance(yVariance),
    _N(_x.cols()),
    _timepoints(timepoints),
    _subjects(subjects),
    _K(K),
    _logLikelihoodGivenZ(_K, _N),
    _Q(_K, _K),
    _exTauN(_K, _K),
    _gamma(_K, _N),
    _xi(_N),
    _pi(_K),
    _logAlpha(_K, _N),
    _logBeta(_K, _N),
    _c(_N),
    _eta(1),
    _ViterbiPath(_N),
    _thresholdOfClusterShrinkage(thresholdOfClusterShrinkage),
    _thresholdOfConvergence(thresholdOfConvergence),
    _iterNum(iterNum),
    _lambda(y.rows()),
    _exPhi(K),
    _exPhiPhiT(K, vector<MatrixXd>(_y.rows()))
{
    this->parseMetadata(_timepoints, _subjects);
    this->initializeParameters();
}//}}}

CTRHMMVariationalEstimator::~CTRHMMVariationalEstimator(){//{{{

}//}}}

void CTRHMMVariationalEstimator::parseMetadata(const vector<double> &timepoints, const vector<string> &subjects){//{{{
    string thisSubjectID(subjects[0]);
    _initPointIdx.push_back(0);
    vector<string> uniqueSubjectID;
    uniqueSubjectID.push_back(subjects[0]);
    for(int n=1; n<subjects.size(); n++){
        if(thisSubjectID != subjects[n]){
            _endPointIdx.push_back(n-1);
            _initPointIdx.push_back(n);
            thisSubjectID = subjects[n];
            uniqueSubjectID.push_back(subjects[n]);
        }
    }
    _endPointIdx.push_back(subjects.size()-1);
    vector<double> nonuniqueDelta;
    for(int s=0; s<uniqueSubjectID.size(); s++){
        int n = _initPointIdx[s] + 1;
        while(n <= _endPointIdx[s]){
            nonuniqueDelta.push_back(timepoints[n] - timepoints[n-1]);
            n++;
        }
        // dummy, never referred
        nonuniqueDelta.push_back(-1);
    }
    for(int i=0; i<nonuniqueDelta.size(); i++){
        // end point
        if(nonuniqueDelta[i] == -1){
            // dummy, never referred
            _deltaIdx.push_back(-1);
        }else{
            auto it = find(_delta.begin(), _delta.end(), nonuniqueDelta[i]);
            // if doesn't exist
            if(it == _delta.end()){
                _delta.push_back(nonuniqueDelta[i]);
                _deltaIdx.push_back(_delta.size()-1);
            }else{
                _deltaIdx.push_back(distance(_delta.begin(), it));
            }
        }
    }

}//}}}

void CTRHMMVariationalEstimator::initializeParameters(){//{{{
    _S = _initPointIdx.size();
    gamma_distribution<double> gamma(1.0, 1.0);
    _gamma = MatrixXd::NullaryExpr(_K, _N, [&](){return gamma(gen);});
    for(int n=0; n<_N; n++){
        _gamma.col(n) = _gamma.col(n) / _gamma.col(n).sum();
    }
    _Q = MatrixXd::NullaryExpr(_K, _K, [&](){return gamma(gen);});
    for(int k=0; k<_K; k++){
        _Q.row(k) = _Q.row(k) / _Q.row(k).sum();
    }
    _Q = _Q - MatrixXd::Identity(_K, _K);
    for(int d=0; d<_delta.size(); d++){
        _PDeltaT.push_back(_Q);
    }
    this->updatePDeltaT();
    for(int d=0; d<_delta.size(); d++){
        _C.push_back(MatrixXd::Zero(_K, _K));
    }
    for(int n=0; n<_N; n++){
        _xi[n] = MatrixXd::Zero(_K, _K);
    }
    _pi = VectorXd::NullaryExpr(_K, [&](){return 1;});
    _pi = _pi / _pi.sum();
    _lambda = VectorXd::NullaryExpr(_y.rows(), [&](){return gamma(gen);});
    normal_distribution<double> normal(0.0, 1.0);
    for(int k=0; k<_K; k++){
        _exPhi[k] = MatrixXd::NullaryExpr(_x.rows(), _y.rows(), [&](){return normal(gen);});
        for(int m=0; m<_y.rows(); m++){
            _exPhiPhiT[k][m] = _exPhi[k].col(m) * _exPhi[k].col(m).transpose() + MatrixXd::Identity(_x.rows(), _x.rows());
        }
    }
    this->updateLambda();
    this->updateLogLikelihoodGivenZ(_x, _y, _yVariance);
}//}}}

void CTRHMMVariationalEstimator::updateExpectations(){//{{{
    this->updateForwardProbs();
    this->updateBackwardProbs();
    this->updateGamma();
    this->updateXi();
    this->updateC();
    this->updateTauAndN();
}//}}}

void CTRHMMVariationalEstimator::updateForwardProbs(){//{{{
    // iteration for each subject
    for(int s=0; s<_S; s++){
        int n = _initPointIdx[s];
        while(n <= _endPointIdx[s]){
            if(n == _initPointIdx[s]){
                _logAlpha.col(n) = _pi.array().log().matrix() + _logLikelihoodGivenZ.col(n);
            }else{
                _logAlpha.col(n) = (_PDeltaT[_deltaIdx[n-1]].transpose() * _logAlpha.col(n-1).array().exp().matrix()).array().log() + _logLikelihoodGivenZ.col(n).array();
            }
            // scaling
            _c(n) = logSumExp(VectorXd(_logAlpha.col(n)));
            _logAlpha.col(n) = _logAlpha.col(n).array() - _c(n);
            n++;
        }
    }
}//}}}

void CTRHMMVariationalEstimator::updateBackwardProbs(){//{{{
    // iteration for each subject
    for(int s=0; s<_S; s++){
        _logBeta.col(_endPointIdx[s]) = VectorXd::Zero(_K);
        int n = _endPointIdx[s] - 1;
        while(n >= _initPointIdx[s]){
            // z_n+1 == k
            for(int k=0; k<_K; k++){
                VectorXd tmp((_PDeltaT[_deltaIdx[n]].row(k).transpose().array().log() + _logBeta.col(n+1).array() + _logLikelihoodGivenZ.col(n+1).array()).matrix());
                _logBeta(k, n) = logSumExp(tmp);
            }
            // scaling
            _logBeta.col(n) = (_logBeta.col(n).array() - _c(n+1));
            n--;
        }
    }
}//}}}

void CTRHMMVariationalEstimator::updateGamma(){//{{{
    _gamma = (_logAlpha + _logBeta).array().exp();
}//}}}

void CTRHMMVariationalEstimator::updateXi(){//{{{
    for(int s=0; s<_S; s++){
        int n = _initPointIdx[s];
        while(n <= _endPointIdx[s]){
            if(n == _endPointIdx[s]){
                // dummy, never referred
                _xi[n] = MatrixXd::Zero(_K, _K);
            }else{
                // z_n+1 == k
                for(int k=0; k<_K; k++){
                    _xi[n].col(k) = _logAlpha.col(n).array() + _logLikelihoodGivenZ(k, n+1) + _logBeta(k, n+1) + _PDeltaT[_deltaIdx[n]].col(k).array().log() - _c(n+1);

                }
                _xi[n] = _xi[n].array().exp().matrix();
            }
            n++;
        }
    }
}//}}}

void CTRHMMVariationalEstimator::updateC(){//{{{
    for(int n=0; n<_N; n++){
        int d = _deltaIdx[n];
        if(d != -1){
            _C[_deltaIdx[n]] += _xi[n];
        }
    }
}//}}}

void CTRHMMVariationalEstimator::updateTauAndN(){//{{{
    MatrixXd A(MatrixXd::Zero(2*_K, 2*_K));
    A.block(0, 0, _K, _K) = _Q;
    A.block(_K, _K, _K, _K) = _Q;
    _exTauN = MatrixXd::Zero(_K, _K);
    for(int d=0; d<_delta.size(); d++){
        for(int i=0; i<_K; i++){
            for(int j=0; j<_K; j++){
                A(i, _K + j) = 1;
                MatrixXd D((A * _delta[d]).exp().block(0, _K, _K, _K).array() / _PDeltaT[d].array());
                if(i == j){
                    // update tau
                    _exTauN(i, i) += (_C[d].array() * D.array()).sum();
                }else{
                    // update n
                    _exTauN(i, j) += (_C[d].array() * (_Q(i, j) * D).array()).sum();
                }
                A(i, _K + j) = 0;
            }
        }
    }
}//}}}

void CTRHMMVariationalEstimator::updateQ(){//{{{
    for(int i=0; i<_K; i++){
        for(int j=0; j<_K; j++){
            if(i != j){
                _Q(i, j) = _exTauN(i, j) / _exTauN(i, i);
            }
        }
        _Q(i, i) = 0;
        _Q(i, i) = - _Q.row(i).sum();
    }
    this->updatePDeltaT();
}//}}}

void CTRHMMVariationalEstimator::updatePDeltaT(){//{{{
    for(int d=0; d<_delta.size(); d++){
        _PDeltaT[d] = (_Q * _delta[d]).exp();
    }
}//}}}

void CTRHMMVariationalEstimator::updatePhi(){//{{{
    _qPhiEntropy = 0;
    for(int m=0; m<_y.rows(); m++){
        MatrixXd lambdaMatrix(MatrixXd::Identity(_x.rows(), _x.rows()) * _lambda(m));
        for(int k=0; k<_K; k++){
            VectorXd LVector(_gamma.row(k).array() / (_yVariance.row(m).array() * _eta));
            MatrixXd L(LVector.asDiagonal());
            MatrixXd precPhi = lambdaMatrix + _x * L * _x.transpose();
            MatrixXd covPhi = precPhi.inverse();
            VectorXd t = VectorXd::Zero(_x.rows());
            t = _x * L * _y.row(m).transpose();
            _exPhi[k].col(m) = covPhi * t;
            _exPhiPhiT[k][m] = _exPhi[k].col(m) * _exPhi[k].col(m).transpose() + covPhi;
            MatrixXd doublePiECovPhi(2 * M_PI * M_E * covPhi);
            _qPhiEntropy += logdet(doublePiECovPhi) / 2;
        }
    }
    this->updateLogLikelihoodGivenZ(_x, _y, _yVariance);
}//}}}

void CTRHMMVariationalEstimator::updateLambda(){//{{{
    for(int m=0; m<_y.rows(); m++){
        double sumPhimTPhim = 0;
        for(int k=0; k<_K; k++){
            sumPhimTPhim += _exPhiPhiT[k][m].trace();
        }
        _lambda(m) = _x.rows() * _K / sumPhimTPhim;
    }
}//}}}

void CTRHMMVariationalEstimator::updateLogLikelihoodGivenZ(const MatrixXd &x, const MatrixXd &y, const MatrixXd &yVariance){//{{{
    int N = _x.cols();
    _residualSum = 0;
    for(int n=0; n<N; n++){
        VectorXd ynVarianceInverse(yVariance.col(n).array().inverse());
        MatrixXd precMatrix(ynVarianceInverse.asDiagonal());
        VectorXd yn(y.col(n));
        VectorXd xn(x.col(n));
        for(int k=0; k<_K; k++){
            double term1 = y.rows() * log(2 * M_PI * _eta) / 2;
            double term2 = yVariance.col(n).array().log().sum() / 2;
            double residual = 0;
            residual += (yn.transpose() * precMatrix * yn)(0, 0);
            residual -= 2 * (yn.transpose() * precMatrix * _exPhi[k].transpose() * xn)(0, 0);
            MatrixXd phiPrecPhiT(MatrixXd::Zero(x.rows(), x.rows()));
            for(int m=0; m<y.rows(); m++){
                MatrixXd S(MatrixXd::Ones(x.rows(), x.rows()) / yVariance(m, n));
                phiPrecPhiT = phiPrecPhiT.array() + S.array() * _exPhiPhiT[k][m].array();
            }
            residual += (xn.transpose() * phiPrecPhiT * xn)(0, 0);
            _logLikelihoodGivenZ(k, n) = - term1 - term2 - residual / (2 * _eta);
            _residualSum += _gamma(k, n) * residual;
        }
    }
}//}}}

void CTRHMMVariationalEstimator::updatePi(){//{{{
    // _pi = VectorXd::Zero(_K);
    // for(int s=0; s<_S; s++){
    //     _pi += _gamma.col(_initPointIdx[s]);
    // }
    _pi = VectorXd::NullaryExpr(_K, [&](){return 1;});
    _pi = _pi / _pi.sum();
}//}}}

void CTRHMMVariationalEstimator::updateEta(){//{{{
    _eta = _residualSum / (_y.rows() * _N);
}//}}}

void CTRHMMVariationalEstimator::calcAndWriteTestLogLikelihood(const MatrixXd &testX, const MatrixXd &testY, const MatrixXd &testYVariance, const vector<double> &testTimepoints, const vector<string> &testSubjects, string outputDirectory){//{{{
    _initPointIdx.clear();
    _endPointIdx.clear();
    _delta.clear();
    _deltaIdx.clear();
    this->parseMetadata(testTimepoints, testSubjects);
    int N = testTimepoints.size();
    _S = _initPointIdx.size();
    _logLikelihoodGivenZ = MatrixXd::Zero(_K, N);
    this->updateLogLikelihoodGivenZ(testX, testY, testYVariance);
    _PDeltaT.clear();
    for(int d=0; d<_delta.size(); d++){
        _PDeltaT.push_back(_Q);
    }
    this->updatePDeltaT();
    _logAlpha = MatrixXd::Zero(_K, N);
    _c = VectorXd::Zero(N);
    this->updateForwardProbs();
    double logLikelihood = _c.sum();
    std::ofstream stream;
    stream.open(outputDirectory + "testLogLikelihood.csv", ios::out);
    stream<<setprecision(numeric_limits<double>::max_digits10);
    stream<<logLikelihood<<endl;
    stream.close();
    _logBeta = MatrixXd::Zero(_K, N);
    this->updateBackwardProbs();
    this->updateGamma();
    MatrixXd logPredictProbForEachZ(_gamma.array().log().matrix() + _logLikelihoodGivenZ);
    VectorXd logPredictProb(N);
    for(int n=0; n<N; n++){
        VectorXd lppfezn(logPredictProbForEachZ.col(n));
        logPredictProb(n) = logSumExp(lppfezn);
    }
    MatrixXd lppMatrix(logPredictProb);
    outputEigenMatrix(lppMatrix, outputDirectory + "testLogPredictProb.csv");
}//}}}

void CTRHMMVariationalEstimator::calculateELBO(){//{{{
    double thisELBO = 0;
    thisELBO += _c.sum();
    thisELBO -= _K * _y.rows() * _x.rows() * log(2 * M_PI) / 2;
    thisELBO += _K * _x.rows() * _lambda.array().log().sum() / 2;
    for(int m=0; m<_y.rows(); m++){
        for(int k=0; k<_K; k++){
            thisELBO -= _lambda(m) * _exPhiPhiT[k][m].trace() / 2;
        }
    }
    thisELBO += _qPhiEntropy;
    _ELBO.push_back(thisELBO);
}//}}}

void CTRHMMVariationalEstimator::findViterbiPath(){//{{{
    _maxPathLogProb = MatrixXd::Zero(_K, _N);
    _maxPath = MatrixXi::Zero(_K, _N);
    for(int s=0; s<_S; s++){
        int n = _initPointIdx[s];
        while(n <= _endPointIdx[s]){
            if(n == _initPointIdx[s]){
                _maxPathLogProb.col(n) = _pi.array().log().matrix() + _logLikelihoodGivenZ.col(n);
            }else{
                MatrixXd pathLogProb(MatrixXd::Zero(_K, _K));
                MatrixXd onesK(MatrixXd::Ones(1, _K));
                pathLogProb = (_maxPathLogProb.col(n-1) * onesK) + _PDeltaT[_deltaIdx[n-1]].array().log().matrix() + (_logLikelihoodGivenZ.col(n) * onesK).transpose();
                int maxIdx;
                for(int k=0; k<_K; k++){
                    _maxPathLogProb(k, n) = pathLogProb.col(k).maxCoeff(&maxIdx);
                    _maxPath(k, n-1) = maxIdx;
                }
            }
            if(n == _endPointIdx[s]){
                int maxIdx;
                _maxPathLogProb.col(n).maxCoeff(&maxIdx);
                for(int k=0; k<_K; k++){
                    _maxPath(k, n) = maxIdx;
                }

            }
            n++;
        }
    }
    // trace back
    int thisPath = 0;
    for(int n=_N-1; n>=0; n--){
        thisPath = _maxPath(thisPath, n);
        _ViterbiPath(n) = thisPath;
    }
}//}}}

void CTRHMMVariationalEstimator::writeParameters(string outputDirectory)const{//{{{
    if(outputDirectory[outputDirectory.size()-1] != '/')outputDirectory.push_back('/');
    outputEigenMatrix(_logAlpha, outputDirectory + "logAlpha.csv");
    outputEigenMatrix(_logBeta, outputDirectory + "logBeta.csv");
    MatrixXd cTemp(_c);
    outputEigenMatrix(cTemp, outputDirectory + "c.csv");
    outputEigenMatrix(_gamma, outputDirectory + "gamma.csv");
    outputEigenMatrix(_exTauN, outputDirectory + "exTauN.csv");
    // for(int d=0; d<_delta.size(); d++){
    //     outputEigenMatrix(_xi[d], outputDirectory + "xi" + to_string(d) + ".csv");
    //     outputEigenMatrix(_PDeltaT[d], outputDirectory + "PDeltaT" + to_string(d) + ".csv");
    // }
    outputEigenMatrix(_logLikelihoodGivenZ, outputDirectory + "logLikelihoodGivenZ.csv");
    outputEigenMatrix(_Q, outputDirectory + "Q.csv");
    MatrixXd piTemp(_pi);
    outputEigenMatrix(piTemp, outputDirectory + "pi.csv");
    outputEigenMatrix(_maxPathLogProb, outputDirectory + "maxPathLogProb.csv");
    outputEigenMatrix(_maxPath, outputDirectory + "maxPath.csv");
    MatrixXi vpTemp(_ViterbiPath);
    outputEigenMatrix(vpTemp, outputDirectory + "ViterbiPath.csv");
    for(int k=0; k<_K; k++){
        outputEigenMatrix(_exPhi[k], outputDirectory + "phi" + to_string(k) + ".csv");
    }
    MatrixXd lambdaTemp(_lambda);
    outputEigenMatrix(lambdaTemp, outputDirectory + "lambda.csv");
    std::ofstream stream;
    stream.open(outputDirectory + "qPhiEntropy.csv", std::ios::out);
    stream<<_qPhiEntropy<<endl;
    stream.close();
    stream.open(outputDirectory + "eta.csv", std::ios::out);
    stream<<_eta<<endl;
    stream.close();
    stream.open(outputDirectory + "rdValue.csv", std::ios::out);
    stream<<rdValue<<endl;
    stream.close();
}//}}}

void CTRHMMVariationalEstimator::writeObjective(string outputDirectory)const{//{{{
    string ELBOFilename;
    ELBOFilename = outputDirectory + "ELBO.csv";
    outputVector(_ELBO, ELBOFilename);
}//}}}

bool CTRHMMVariationalEstimator::isConvergence()const{//{{{
    if(_ELBO.size()<2)return false;
    double thisELBO = _ELBO[_ELBO.size()-1];
    double prevELBO = _ELBO[_ELBO.size()-2];
    double rate = (thisELBO - prevELBO) / abs(prevELBO);
    if(rate < _thresholdOfConvergence){
        return true;
    }else{
        return false;
    }
}//}}}

void CTRHMMVariationalEstimator::shrinkCluster(int k){//{{{
    eraseVectorElement(_exPhi, k);
    eraseVectorElement(_exPhiPhiT, k);
    removeRow(_logAlpha, k);
    removeRow(_logBeta, k);
    removeRow(_gamma, k);
    for(int n=0; n<_xi.size(); n++){
        removeRow(_xi[n], k);
        removeColumn(_xi[n], k);
    }
    for(int d=0; d<_C.size(); d++){
        removeRow(_C[d], k);
        removeColumn(_C[d], k);
    }
    removeRow(_exTauN, k);
    removeColumn(_exTauN, k);
    removeRow(_Q, k);
    removeColumn(_Q, k);
    for(int d=0; d<_PDeltaT.size(); d++){
        removeRow(_PDeltaT[d], k);
        removeColumn(_PDeltaT[d], k);
    }
    removeRow(_logLikelihoodGivenZ, k);
    removeElement(_pi, k);
    _K--;
    this->updateQ();
    this->updatePi();
    this->updateForwardProbs();
    this->updateBackwardProbs();
    this->updateGamma();
    this->updateXi();
    this->updateC();
    this->updateTauAndN();
    this->updatePhi();
    this->updateLambda();
    this->calculateELBO();
}//}}}

bool CTRHMMVariationalEstimator::checkNonRelevantCluster(){//{{{
    // considering shifts of indices
    VectorXd gammaRowwiseSum(_gamma.rowwise().sum());
    bool isShrinked = false;
    for(int k=_K-1; k>=0; k--){
        if(gammaRowwiseSum(k) <= _x.rows()){
            this->shrinkCluster(k);
            isShrinked = true;
            break;
        }
    }
    return isShrinked;
}//}}}

void CTRHMMVariationalEstimator::runIteraions(){//{{{
    int c = 0;
    bool isShrinked = false;
    while(1){
        // this->writeParameters("./Iter" + to_string(c));
        this->updateForwardProbs();
        this->updateBackwardProbs();
        this->updateGamma();
        this->updateXi();
        this->updateC();
        this->updateTauAndN();
        this->calculateELBO();
        if(!std::isfinite(_ELBO[_ELBO.size()-1]))break;
        if(std::isnan(_ELBO[_ELBO.size()-1]))break;
        c++;
        if(c > _iterNum || (this->isConvergence() && !isShrinked)){
            cout<<endl;
            break;
        }
        cout<< '\r'<< "Loop: #"<<c<< string(20, ' ');
        flush(cout);
        isShrinked = this->checkNonRelevantCluster();
        this->updateQ();
        this->updateEta();
        this->updatePhi();
        this->updateLambda();
        // this->updatePi();
    }
    this->findViterbiPath();
}//}}}
