#include "dual.h"
#include "gtest/gtest.h"

namespace
{
    TEST(Dual, PowMultiplication)
    {
        cpp_nn::Dual x(5, 1); // Seeding x, so that taking derivative w.r.t x
        cpp_nn::Dual y(6, 0);

        auto w1 = pow(x, 2);

        cpp_nn::Dual f = y * w1;
    
        std::cout << "f: " << f.getDerivative() << " "
                << "x: " << x.getDerivative() << " "
                << "w1: " << w1.getDerivative() << " " 
                << "y: " << y.getDerivative() << " "
                << std::endl;
        
        EXPECT_EQ(60, f.getDerivative());
        EXPECT_EQ(10, w1.getDerivative());
        EXPECT_EQ(1, x.getDerivative());
        EXPECT_EQ(0, y.getDerivative());
    }
}