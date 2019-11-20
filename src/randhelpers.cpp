#include "randhelpers.h"

namespace neural_network
{
    double randMToN(double M, double N)
    {
        return M + (rand() / ( RAND_MAX / (N-M) ) ) ;  
    }
    
    int randMinToMax(int min, int max)
    {
        auto output = min + (rand() % static_cast<int>(max - min + 1));
        return output;
    }
}