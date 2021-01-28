//
// utils.hpp
//
// Copyright (c) 2018 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//

#ifndef UTILS
#define UTILS

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
#include <boost/lexical_cast.hpp>
#include<Eigen/Dense>
#include<Eigen/Core>
#include<Eigen/LU>
#include <Eigen/Cholesky>

template<class T>
void readCSV(std::string filename, std::vector<std::vector<T> > &matrix, std::vector<std::string> *rowNamesPointer=nullptr, std::vector<std::string> *columnNamesPointer=nullptr){
    matrix.clear();
    if(rowNamesPointer != nullptr){
        (*rowNamesPointer).clear();
    }
    if(columnNamesPointer != nullptr){
        (*columnNamesPointer).clear();
    }
    std::ifstream inputText(filename);
    if(!inputText){
        std::cout<<"Cannot open Csvfile";
        exit(1);
    }
    std::string str;
    // row
    for(int i=0; std::getline(inputText,str); i++){
        std::vector<T> vec;
        std::string token;
        std::istringstream stream(str);

        //column name
        if(i==0 && columnNamesPointer!=nullptr){
            //column
            for(int j=0; std::getline(stream,token,','); j++){
                if(j==0 && rowNamesPointer!=nullptr)continue;
                (*columnNamesPointer).push_back(token);
            }
            continue;
        }
        //column
        for(int j=0; std::getline(stream,token,','); j++){
            if(j==0 && rowNamesPointer!=nullptr)(*rowNamesPointer).push_back(token);
            else{
                try{
                    vec.push_back(boost::lexical_cast<T>(token));
                }catch(...){
                    std::cout<<"row:"<<i<<" column:"<<j<<" value:"<<token<<" error";
                    exit(0);
                }
            }
        }
        matrix.push_back(vec);
    }
}

template<typename T>
void parseCSV2Eigen(std::string filename, Eigen::Matrix<T, Eigen::Dynamic, 1> &eigenVector, std::vector<std::string> *rowNamesPointer=nullptr, std::vector<std::string> *columnNamesPointer=nullptr){
    std::vector<std::vector<T> > matrix;
    readCSV<T>(filename, matrix, rowNamesPointer, columnNamesPointer);
    eigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>(matrix.size());
    for (int i = 0; i < matrix.size(); i++){
        eigenVector(i) = matrix[i][0];
    }
}

template<typename T>
void parseCSV2Eigen(std::string filename, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigenMatrix, std::vector<std::string> *rowNamesPointer=nullptr, std::vector<std::string> *columnNamesPointer=nullptr){
    std::vector<std::vector<T> > matrix;
    readCSV<T>(filename, matrix, rowNamesPointer, columnNamesPointer);
    eigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(matrix.size(), matrix[0].size());
    for (int i = 0; i < matrix.size(); i++){
        eigenMatrix.row(i) = Eigen::Matrix<T, Eigen::Dynamic, 1>::Map(&matrix[i][0], matrix[0].size());
    }
}

template<class T>
void convertUnique(const std::vector<T> &v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

template<typename T>
double logdet(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix){
    return 2.0 * Eigen::LLT<Eigen::MatrixXd>(matrix).matrixL().toDenseMatrix().diagonal().array().log().sum();
}

template<typename T>
double logSumExp(const Eigen::Matrix<T, Eigen::Dynamic, 1> &logVector){
    double constant(logVector.maxCoeff());
    double sumexp((logVector.array() - constant).array().exp().sum());
    return log(sumexp) + constant;
}

template<typename T>
Eigen::VectorXd normalizeWithLogSumExp(const Eigen::Matrix<T, Eigen::Dynamic, 1> &logVector){
    Eigen::VectorXd resVector = (logVector.array() - logVector.maxCoeff()).array().exp();
    return resVector/resVector.sum();
}

template<typename T>
void eraseVectorElement(std::vector<T> &vector, unsigned int idx){
    vector.erase(vector.begin() + idx);
}

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

template<typename T>
void outputEigenMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix, std::string filename){
    std::ofstream stream(filename);
    stream<<matrix.format(CSVFormat);
    stream.close();
 }

template<typename T>
void outputVectorEigenVector(const std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> &vectorVector, std::string filename){
    std::ofstream stream(filename);
    for(int i=0; i<vectorVector.size(); i++){
        // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempMatrix(vectorVector[i].transpose());
        // stream<<tempMatrix.format(CSVFormat);
        stream<<vectorVector[i].transpose().format(CSVFormat)<<std::endl;
        // std::cout<<tempMatrix.format(CSVFormat);
    }
    stream.close();
 }

template<typename T>
void outputEigenMatrixForDebug(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix, std::string msg){
    std::ofstream stream;
    stream.open("./check", std::ios::app);
    stream<<msg<<std::endl;
    stream<<matrix.format(CSVFormat);
    stream<<std::endl;
    stream.close();
}

template<typename T>
void outputVector(const std::vector<T> &vector, std::string filename){
    std::ofstream stream;
    stream.open(filename, std::ios::out);
    stream<<std::setprecision(std::numeric_limits<double>::max_digits10);
    for(int i=0;i<vector.size();i++){
        stream<<vector[i];
        stream<<std::endl;
    }
    stream.close();
}


#endif
