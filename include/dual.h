#ifndef DUAL
#define DUAL

#include <iostream>
#include <cmath>

namespace cpp_nn
{
    class Dual
    {
        private:
            double val;
            double der;
        public:
            Dual();
            Dual(double val);
            Dual(double val, double derivative);

            double getDerivative() const;
            void setDerivative(double derivative);

            friend Dual operator+(const Dual& lhs, const Dual& rhs);
            friend Dual operator-(const Dual& lhs, const Dual& rhs);
            friend Dual operator*(const Dual& lhs, const Dual& rhs);
            friend Dual operator/(const Dual& lhs, const Dual& rhs);

            friend std::ostream& operator<<(std::ostream& os, const Dual& d);

            friend Dual sin(Dual d);
            friend Dual cos(Dual d);
            friend Dual exp(Dual d);
            friend Dual log(Dual d);
            friend Dual abs(Dual d);
            friend Dual pow(Dual d, double p);
    };
}

#endif