//
//  main.cpp
//  mathpls
//
//  Created by Ninter6 on 2023/2/10.
//

#include <iostream>
#include <unistd.h>
//#include <cmath>

#include "mathpls.h"

using namespace std;
using namespace mathpls;

int main(int argc, const char * argv[]) {
    vec4 v(1, 0, 0, 1);
    v = rotate(PI/2, PI, PI/2, EARS::yxy) * v;
    cout<<v.r<<" "<<v.g<<" "<<v.b<<endl;
    
//    for(int i = 0; i < 360; i++){
//        long double p = mathpls::radians<long double>(i);
//        cout<<i<<"度: "<<sin(p)<<" "<<cos(p)<<" "<<tan(p)<<" "<<cot(p)<<" "<<sec(p)<<" "<<csc(p)<<endl;
//    }
    return 0;
}
