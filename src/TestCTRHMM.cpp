//
// TestCTRHMM.cpp
//
// Copyright (c) 2020 Shion Hosoda
//
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php
//

//include{{{
#include "include/CTRHMMVETester.hpp"

//}}}

using namespace std;
using namespace Eigen;

int main(int argc, char *argv[]){
    MatrixXd x, y, yVariance;
    parseCSV2Eigen("./testcase/input/x.csv", x);
    parseCSV2Eigen("./testcase/input/y.csv", y);
    parseCSV2Eigen("./testcase/input/y_variance.csv", yVariance);
    vector<double> timepoints{0.0, 1.4, 2.4, 0.0, 1.0};
    vector<string> subjects{"subject1", "subject1", "subject1", "subject2", "subject2"};
    CTRHMMVETester Tester(x, y, yVariance, timepoints, subjects, 2, 0, 0, 100);
    Tester.testAll();
    return 0;
}
