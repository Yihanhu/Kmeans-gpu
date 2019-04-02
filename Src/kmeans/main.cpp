//  MNIST.h
/*************************************************************************
 * MNIST Dataset Loader
 *------------------------------------------------------------------------
 * Copyright (c) 2016 Peter Baumann
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would
 *    be appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not
 *    be misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source
 *    distribution.
 *
 *************************************************************************/

// CLANG / GCC compiler flags: -std=c++14 -Os

#include <iostream>
#include <chrono>

#include "MNIST.h"

using namespace std;
using namespace chrono;

int main() {
    
//    const steady_clock::time_point start = steady_clock::now();
    MNIST mnist = MNIST("/Users/peter/Downloads/");
//    const steady_clock::time_point stop = steady_clock::now();
//    cout << "Exec. Time: " <<duration_cast<milliseconds>(stop - start).count() <<" ms" <<endl;
    
    mnist.testPrintout(40, 60);
    
    return 0;
}
