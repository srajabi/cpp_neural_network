#include "variable.h"
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <ctime>

namespace neural_network
{
    double randMToN(double M, double N)
    {
        return M + (rand() / ( RAND_MAX / (N-M) ) ) ;  
    }

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

    void Variable::Backward()
    {
        // TODO
    }

    void Variable::Forward()
    {
        // TODO
    }

    double Variable::getGradient()
    {
        /*
        std::cout << "getGradient(): gradient: "
                    << this->gradient << " "
                    << this
                    << std::endl;
                    */

        if(isnan(this->gradient))
        {
            double sum = 0;
            for(auto i : this->children)
            {
                /*
                std::cout << "pre getGradient i " << " ";
                std::cout << i.variable << " " << i.derivative() << std::endl;
                */
                sum += i.GetGradient();
                /*
                std::cout << "post getGradient i " << " ";
                std::cout << i.variable << " " << i.derivative() << std::endl;
                std::cout << sum << std::endl;
                */
            }

            this->gradient = sum;
        }
        return this->gradient;
    }

    void Variable::setGradient(double gradient)
    {
        this->gradient = gradient;
    }

    double Variable::getValue()
    {
        this->value = this->function();
        return this->value;
    }

    void Variable::setValue(double value)
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
                return a->getValue() + b->getValue();
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
                return a->getValue() - b->getValue();
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
                return a->getValue() * b->getValue();
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
/*
    std::shared_ptr<Variable> operator/(
        std::shared_ptr<Variable> lhs,
        std::shared_ptr<Variable> rhs)
    {
        std::shared_ptr<Variable> out(
            new Variable(lhs->value / rhs->value));

        lhs->children.push_back({out, 1 / rhs->value});
        rhs->children.push_back({out, -1 / ::pow(rhs->value, 2)});

        return out;
    }
*/
    std::ostream& operator<<(
        std::ostream& os,
        const std::shared_ptr<Variable> v)
    {
        os << "(" << v->value << "," << v->gradient << ")" << v.get();
        return os;
    }

    std::shared_ptr<Variable> sin(std::shared_ptr<Variable> v)
    {
        std::function<double()> sin_fn = [=]()
        {
            return ::sin(v->getValue());
        };
        std::shared_ptr<Variable> out(new Variable(sin_fn));

        std::function<double()> derivative = [=]()
        {
            return ::cos(v->value);
        };

        v->children.push_back({out, derivative});

        return out;
    }
/*
    std::shared_ptr<Variable> cos(std::shared_ptr<Variable> v)
    {
        std::shared_ptr<Variable> out(new Variable(::cos(v->value)));

        v->children.push_back({out, -::sin(v->value)});

        return out;
    }

    std::shared_ptr<Variable> exp(std::shared_ptr<Variable> v)
    {
        std::shared_ptr<Variable> out(new Variable(::exp(v->value)));

        v->children.push_back({out, ::exp(v->value)});

        return out;
    }

    std::shared_ptr<Variable> log(std::shared_ptr<Variable> v)
    {
        std::shared_ptr<Variable> out(new Variable(::log(v->value)));

        v->children.push_back({out, 1 / v->value});

        return out;
    }

    std::shared_ptr<Variable> abs(std::shared_ptr<Variable> v)
    {
        std::shared_ptr<Variable> out(new Variable(::abs(v->value)));

        double sign = v->value == 0 ? 0 : v->value / ::abs(v->value);
        v->children.push_back({out, sign});

        return out;
    }

    std::shared_ptr<Variable> pow(std::shared_ptr<Variable> v, double p)
    {
        std::shared_ptr<Variable> out(new Variable(::pow(v->value, p)));

        v->children.push_back({out, p * ::pow(v->value, p-1)});

        return out;
    }
*/
    std::shared_ptr<Variable> sigmoid(std::shared_ptr<Variable> v)
    {
        std::function<double()> sigmoid_fn = [=]()
        {
            return 1 / (1 + ::exp(-v->getValue()));
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
/*
    std::shared_ptr<Variable> relu(std::shared_ptr<Variable> v)
    {
        std::shared_ptr<Variable> out(new Variable(::fmax(0, v->value)));

        double derivative = v->value > 0 ? 1 : 0;

        v->children.push_back({out, derivative});

        return out;
    }
*/
}