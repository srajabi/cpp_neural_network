#include "variable.h"
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include "randhelpers.h"

namespace neural_network
{
    Variable::Variable()
    {
        this->value = randMToN(-1.5, 1.5);
        this->gradient = NAN;

        this->function = [=]()
        {
            return this->value;
        };
    }

    Variable::Variable(double val)
    {
        this->value = val;
        this->gradient = NAN;

        this->function = [=]()
        {
            return this->value;
        };
    }

    Variable::Variable(std::function<double()> fn)
    {
        this->function = fn;
        this->gradient = NAN;
    }

    void Variable::ClearGradient()
    {
        this->gradient = NAN;
        for(auto i : this->children)
        {
            i.variable->ClearGradient();
        }
    }
    
    double Variable::Gradient()
    {
        if(isnan(this->gradient))
        {
            double sum = 0;
            for(auto i : this->children)
            {
                sum += i.Gradient();
            }

            this->gradient = sum;
        }
        return this->gradient;
    }

    void Variable::Gradient(double gradient)
    {
        this->gradient = gradient;
    }

    double Variable::Value()
    {
        this->value = this->function();
        return this->value;
    }

    void Variable::Value(double value)
    {
        this->value = value;
    }

    std::shared_ptr<Variable> operator+(
        std::shared_ptr<Variable> lhs,
        std::shared_ptr<Variable> rhs)
    {
        std::function<double()> fn = std::bind(
            [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b)
            {
                return a->Value() + b->Value();
            }, lhs, rhs);

        std::shared_ptr<Variable> out(new Variable(fn));

        std::function<double()> derivative = []()
        {
            return 1.0;
        };

        lhs->children.push_back({out, derivative});
        rhs->children.push_back({out, derivative});

        return out;
    }

    std::shared_ptr<Variable> operator-(
        std::shared_ptr<Variable> lhs,
        std::shared_ptr<Variable> rhs)
    {
        std::function<double()> fn = std::bind(
            [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b)
            {
                return a->Value() - b->Value();
            }, lhs, rhs);

        std::shared_ptr<Variable> out(
            new Variable(fn));

        std::function<double()> derivative = []()
        {
            return 1.0;
        };

        lhs->children.push_back({out, derivative});
        rhs->children.push_back({out, derivative});

        return out;
    }

    std::shared_ptr<Variable> operator*(
        std::shared_ptr<Variable> lhs,
        std::shared_ptr<Variable> rhs)
    {
        std::function<double()> fn = std::bind(
            [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b)
            {
                return a->Value() * b->Value();
            }, lhs, rhs);

        std::shared_ptr<Variable> out(
            new Variable(fn));

        std::function<double()> lhs_derivative = [=]()
        {
            return rhs->value;
        };

        std::function<double()> rhs_derivative = [=]()
        {
            return lhs->value;
        };

        lhs->children.push_back({out, lhs_derivative});
        rhs->children.push_back({out, rhs_derivative});

        return out;
    }

    std::shared_ptr<Variable> operator/(
        std::shared_ptr<Variable> lhs,
        std::shared_ptr<Variable> rhs)
    {
        std::function<double()> fn = std::bind(
            [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b)
            {
                return a->Value() / b->Value();
            }, lhs, rhs);

        std::shared_ptr<Variable> out(
            new Variable(fn));

        std::function<double()> lhs_derivative = [=]()
        {
            return 1 / rhs->value;
        };

        std::function<double()> rhs_derivative = [=]()
        {
            return -1 / ::pow(rhs->value, 2);
        };
        
        lhs->children.push_back({out, lhs_derivative});
        rhs->children.push_back({out, rhs_derivative});

        return out;
    }

    std::ostream& operator<<(
        std::ostream& os,
        const std::shared_ptr<Variable> v)
    {
        os << "(" << v->value << "," << v->gradient << ")";
        return os;
    }

    std::shared_ptr<Variable> sin(std::shared_ptr<Variable> v)
    {
        std::function<double()> sin_fn = [=]()
        {
            return ::sin(v->Value());
        };
        std::shared_ptr<Variable> out(new Variable(sin_fn));

        std::function<double()> derivative = [=]()
        {
            return ::cos(v->value);
        };

        v->children.push_back({out, derivative});

        return out;
    }

    std::shared_ptr<Variable> cos(std::shared_ptr<Variable> v)
    {
        std::function<double()> fn = [=]()
        {
            return ::cos(v->Value());
        };
        std::shared_ptr<Variable> out(new Variable(fn));

        std::function<double()> derivative = [=]()
        {
            return -::sin(v->value);
        };

        v->children.push_back({out, derivative});

        return out;
    }

    std::shared_ptr<Variable> exp(std::shared_ptr<Variable> v)
    {
        std::function<double()> fn = [=]()
        {
            return ::exp(v->Value());
        };
        std::shared_ptr<Variable> out(new Variable(fn));

        std::function<double()> derivative = [=]()
        {
            return ::exp(v->value);
        };

        v->children.push_back({out, derivative});

        return out;
    }

    std::shared_ptr<Variable> log(std::shared_ptr<Variable> v)
    {
        std::function<double()> fn = [=]()
        {
            return ::log(v->Value());
        };
        std::shared_ptr<Variable> out(new Variable(fn));

        std::function<double()> derivative = [=]()
        {
            return 1 / v->value;
        };

        v->children.push_back({out, derivative});

        return out;
    }

    std::shared_ptr<Variable> abs(std::shared_ptr<Variable> v)
    {
        std::function<double()> fn = [=]()
        {
            return ::abs(v->Value());
        };
        std::shared_ptr<Variable> out(new Variable(fn));

        std::function<double()> derivative = [=]()
        {
            double sign = v->value == 0 ? 0 : v->value / ::abs(v->value);
            return sign;
        };

        v->children.push_back({out, derivative});

        return out;
    }

    std::shared_ptr<Variable> pow(std::shared_ptr<Variable> v, double p)
    {
        std::function<double()> fn = [=]()
        {
            return ::pow(v->Value(), p);
        };
        std::shared_ptr<Variable> out(new Variable(fn));

        std::function<double()> derivative = [=]()
        {
            return p * ::pow(v->value, p-1);
        };

        v->children.push_back({out, derivative});

        return out;
    }

    std::shared_ptr<Variable> sigmoid(std::shared_ptr<Variable> v)
    {
        std::function<double()> sigmoid_fn = [=]()
        {
            return 1 / (1 + ::exp(-v->Value()));
        };
        std::shared_ptr<Variable> out(new Variable(sigmoid_fn));

        std::function<double()> derivative = [=]()
        {
            double sigmoid = out->value;
            return sigmoid * (1 - sigmoid);
        };

        v->children.push_back({out, derivative});

        return out;
    }

    std::shared_ptr<Variable> relu(std::shared_ptr<Variable> v)
    {
        std::function<double()> fn = [=]()
        {
            return ::fmax(0, v->Value());
        };
        std::shared_ptr<Variable> out(new Variable(fn));

        std::function<double()> derivative = [=]()
        {
            return v->value > 0 ? 1 : 0;
        };

        v->children.push_back({out, derivative});

        return out;
    }
}