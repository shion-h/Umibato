//
// CTRHMM.cpp
//
// Copyright (c) 2020 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//

//include{{{
#include <stdlib.h>
#include <boost/program_options.hpp>
#include "include/CTRHMMVariationalEstimator.hpp"
//}}}

using namespace std;
using namespace Eigen;

MatrixXd attachBiasTerm(const MatrixXd &x){//{{{
    MatrixXd xWithBias(MatrixXd::Ones(x.rows()+1, x.cols()));
    xWithBias.block(0, 0, x.rows(), x.cols()) = x;
    return xWithBias;
}//}}}

void drawOutNeededColumns(const vector<vector<string> > &metadataMatrix, const vector<string> &metadataColumnNames, vector<double> &timepoints, vector<string> &subjects){//{{{
    for(int i=0; i<metadataMatrix.size(); i++){
        for(int j=0; j<metadataMatrix[i].size(); j++){
            if(metadataColumnNames[j]=="timepoint"){
                timepoints.push_back(stod(metadataMatrix[i][j]));
            }else if(metadataColumnNames[j]=="subjectID"){
                subjects.push_back(metadataMatrix[i][j]);
            }
        }
    }
}//}}}

int main(int argc, char *argv[]){

    //Options{{{
    boost::program_options::options_description opt("Options");
    opt.add_options()
    ("help,h", "show help")
    ("output,o", boost::program_options::value<string>()->default_value("./"), "Directory name for output")
    ("threshold-convergence,a", boost::program_options::value<double>()->default_value(0.0), "Threshold of convergence")
    ("threshold-cluster-shrinkage,b", boost::program_options::value<double>()->default_value(0.0), "Threshold of shrinkage of pi")
    ("iteration-number,n", boost::program_options::value<int>()->default_value(100), "The number of iteration")
    ("number-of-clusters,k", boost::program_options::value<int>()->default_value(10), "The initial number of clusters")
    ("test-x,x", boost::program_options::value<string>()->default_value(""), "Test x data filename (optional)")
    ("test-y,y", boost::program_options::value<string>()->default_value(""), "Test y data filename (optional)")
    ("test-y-variance,v", boost::program_options::value<string>()->default_value(""), "Test y variance data filename (optional)")
    ("test-metadata,m", boost::program_options::value<string>()->default_value(""), "Test metadata filename (optional)")
    ;

    boost::program_options::positional_options_description pd;
    pd.add("xfile", 1);
    pd.add("yfile", 1);
    pd.add("yvariancefile", 1);
    pd.add("metadatafile", 1);

    boost::program_options::options_description hidden("hidden");
    hidden.add_options()
        ("xfile", boost::program_options::value<string>(), "hidden")
        ("yfile", boost::program_options::value<string>(), "hidden")
        ("yvariancefile", boost::program_options::value<string>(), "hidden")
        ("metadatafile", boost::program_options::value<string>(), "hidden")
        ;
    boost::program_options::options_description cmdline_options;
    cmdline_options.add(opt).add(hidden);

    boost::program_options::variables_map vm;
    try{
        boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(cmdline_options).positional(pd).run(), vm);
    }catch(const boost::program_options::error_with_option_name& e){
        cout<<e.what()<<endl;
    }
    boost::program_options::notify(vm);
    string xFilename, yFilename, yVarianceFilename, metadataFilename;
    string testXFilename, testYFilename, testYVarianceFilename, testMetadataFilename;
    double thresholdOfConvergence, thresholdOfClusterShrinkage;
    int iterNum;
    int K;
    string outputDirectory;
    if (vm.count("help") || !vm.count("xfile") || !vm.count("yfile") || !vm.count("yvariancefile")  || !vm.count("metadatafile")){
        cout<<"Usage:\n CTRHMM [x file] [y file] [y variance file] [metadata file] [-options] "<<endl;
        cout<<endl;
        cout<<opt<<endl;
        exit(1);
    }else{
        xFilename = vm["xfile"].as<string>();
        yFilename = vm["yfile"].as<string>();
        yVarianceFilename = vm["yvariancefile"].as<string>();
        metadataFilename = vm["metadatafile"].as<string>();
        outputDirectory = vm["output"].as<string>();
        if(outputDirectory[outputDirectory.size()-1] != '/')outputDirectory.push_back('/');
        thresholdOfConvergence = vm["threshold-convergence"].as<double>();
        thresholdOfClusterShrinkage = vm["threshold-cluster-shrinkage"].as<double>();
        iterNum = vm["iteration-number"].as<int>();
        K = vm["number-of-clusters"].as<int>();
        testXFilename = vm["test-x"].as<string>();
        testYFilename = vm["test-y"].as<string>();
        testYVarianceFilename = vm["test-y-variance"].as<string>();
        testMetadataFilename = vm["test-metadata"].as<string>();
    }
    //}}}

    //process metadata{{{
    vector<string> metadataColumnNames;
    vector<vector<string> > metadataMatrix;
    readCSV<string>(metadataFilename, metadataMatrix, nullptr, &metadataColumnNames);
    vector<double> timepoints;
    vector<string> subjects;
    drawOutNeededColumns(metadataMatrix, metadataColumnNames, timepoints, subjects);
    MatrixXd x, y, yVariance;
    vector<string> rowNames, columnNames;
    parseCSV2Eigen(xFilename, x, &rowNames, &columnNames);
    x = attachBiasTerm(x);
    parseCSV2Eigen(yFilename, y, &rowNames, &columnNames);
    parseCSV2Eigen(yVarianceFilename, yVariance, &rowNames, &columnNames);
    //}}}

//estimation{{{
    CTRHMMVariationalEstimator *estimator;

    estimator = new CTRHMMVariationalEstimator(x, y, yVariance, timepoints, subjects, K, thresholdOfClusterShrinkage, thresholdOfConvergence, iterNum);
    estimator->runIteraions();
    estimator->writeParameters(outputDirectory);
    estimator->writeObjective(outputDirectory);

    //{{{test
    if(testXFilename != "" && testYFilename != "" && testYVarianceFilename != "" && testMetadataFilename != ""){
        vector<string> testMetadataColumnNames;
        vector<vector<string> > testMetadataMatrix;
        readCSV<string>(testMetadataFilename, testMetadataMatrix, nullptr, &testMetadataColumnNames);
        vector<double> testTimepoints;
        vector<string> testSubjects;
        drawOutNeededColumns(testMetadataMatrix, testMetadataColumnNames, testTimepoints, testSubjects);
        MatrixXd testX, testY, testYVariance;
        parseCSV2Eigen(testXFilename, testX, &rowNames, &columnNames);
        testX = attachBiasTerm(testX);
        parseCSV2Eigen(testYFilename, testY, &rowNames, &columnNames);
        parseCSV2Eigen(testYVarianceFilename, testYVariance, &rowNames, &columnNames);
        estimator->calcAndWriteTestLogLikelihood(testX, testY, testYVariance, testTimepoints, testSubjects, outputDirectory);
    }
    //}}}

    delete estimator;
//}}}
    return 0;
}
