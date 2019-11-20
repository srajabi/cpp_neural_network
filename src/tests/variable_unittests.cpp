#include "variable.h"
#include "gtest/gtest.h"

using namespace neural_network;
namespace
{
    TEST(Variable, SinMultiplyAdd)
    {
        std::shared_ptr<Variable> x(new Variable(0.5));
        std::shared_ptr<Variable> y(new Variable(4.2));

        auto a = sin(x);
        auto b = x * y;
        auto c = a + b;

        std::cout << c->Value() << std::endl;

        std::cout << "x, y, a, b, c: ";
        std::cout << x << ", ";
        std::cout << y << ", ";
        std::cout << a << ", ";
        std::cout << b << ", ";
        std::cout << c << ", " << std::endl;

        std::cout << " Setting gradient. " << std::endl;
        c->Gradient(1);
        std::cout << "x grad: " << x->Gradient() << std::endl;
        std::cout << "y grad: " << y->Gradient() << std::endl;

        std::cout << "c: " << c << std::endl;
        std::cout << "x: " << x << std::endl;
        std::cout << "y: " << y << std::endl;

        EXPECT_FLOAT_EQ(5.0775824, x->Gradient());
        EXPECT_FLOAT_EQ(0.5, y->Gradient());
        EXPECT_FLOAT_EQ(1, c->Gradient());

        EXPECT_FLOAT_EQ(2.5794256, c->Value());
        EXPECT_FLOAT_EQ(0.5, x->Value());
        EXPECT_FLOAT_EQ(4.2, y->Value());
    }
}