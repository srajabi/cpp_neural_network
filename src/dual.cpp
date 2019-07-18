#include "dual.h"

namespace cpp_nn
{
    Dual::Dual()
    {
        this->val = 0;
        this->der = 0;
    }

    Dual::Dual(double val)
    {
        this->val = val;
        this->der = 0;
    }

    Dual::Dual(double val, double der)
    {
        this->val = val;
        this->der = der;
    }

    double Dual::getDerivative() const
    {
        return this->der;
    }

    void Dual::setDerivative(double der)
    {
        this->der = der;
    }

    Dual operator+(const Dual& u, const Dual& v)
    {
        return Dual(u.val+v.val,
                    u.der+v.der);
    }

    Dual operator-(const Dual& u, const Dual& v)
    {
        return Dual(u.val-v.val,
                    u.der-v.der);
    }

    Dual operator*(const Dual& u, const Dual& v)
    {
        return Dual(u.val/v.val,
                    (u.der*v.val+u.val*v.der));
    }

    Dual operator/(const Dual& u, const Dual& v)
    {
        return Dual(u.val/v.val,
                    (u.der*v.val-u.val*v.der)/(v.val*v.val));
    }

    std::ostream& operator<<(std::ostream& os, const Dual& a)
    {
        os << a.val;
        return os;
    }

    Dual sin(Dual d)
    {
        return Dual(::sin(d.val),
                    d.der * ::cos(d.val));
    }

    Dual cos(Dual d)
    {
        return Dual(::cos(d.val),
                    -d.der * ::sin(d.val));
    }

    Dual exp(Dual d)
    {
        return Dual(::exp(d.val),
                    d.der * ::exp(d.val));
    }

    Dual log(Dual d)
    {
        return Dual(::log(d.val),
                    d.der/d.val);
    }

    Dual abs(Dual d)
    {
        int sign = d.val == 0 ? 0 : d.val / ::abs(d.val);
        return Dual(::abs(d.val),
                    d.der * sign);
    }

    Dual pow(Dual d, double p)
    {
        return Dual(::pow(d.val, p),
                    p * d.der * ::pow(d.val, p-1));
    }
}