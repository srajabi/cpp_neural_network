#include <cmath>

namespace cpp_nn
{
    float relu(float in)
    {
        return fmax(0, in);
    }
}