#pragma once

#include <vector>
#include <ostream>
#include <functional>

namespace neural_network
{
    class Variable
    {
        public:
            Variable();
            Variable(double val);
            Variable(std::function<double()> fn);

            void ClearGradient();

            void Gradient(double gradient);
            double Gradient();

            void Value(double value);
            double Value();

            friend std::shared_ptr<Variable> operator+(
                std::shared_ptr<Variable> lhs,
                std::shared_ptr<Variable> rhs);
            friend std::shared_ptr<Variable> operator-(
                std::shared_ptr<Variable> lhs,
                std::shared_ptr<Variable> rhs);
            friend std::shared_ptr<Variable> operator*(
                std::shared_ptr<Variable> lhs,
                std::shared_ptr<Variable> rhs);
            friend std::shared_ptr<Variable> operator/(
                std::shared_ptr<Variable> lhs,
                std::shared_ptr<Variable> rhs);

            friend std::ostream& operator<<(std::ostream& os,
                                            const std::shared_ptr<Variable> v);

            friend std::shared_ptr<Variable> sin(std::shared_ptr<Variable> v);
            friend std::shared_ptr<Variable> cos(std::shared_ptr<Variable> v);
            friend std::shared_ptr<Variable> exp(std::shared_ptr<Variable> v);
            friend std::shared_ptr<Variable> log(std::shared_ptr<Variable> v);
            friend std::shared_ptr<Variable> abs(std::shared_ptr<Variable> v);
            friend std::shared_ptr<Variable> pow(std::shared_ptr<Variable> v, double p);
            friend std::shared_ptr<Variable> sigmoid(std::shared_ptr<Variable> v);
            friend std::shared_ptr<Variable> relu(std::shared_ptr<Variable> v);

        protected:
            double value;
            double gradient;
            std::function<double()> function;
            
            struct DerivativeVariableTuple
            {
                std::shared_ptr<Variable> variable;
                std::function<double()> derivative;

                double Gradient()
                {
                    return variable->Gradient() * derivative();
                }
            };
            
            std::vector<DerivativeVariableTuple> children;
            std::vector<Variable> parents;
    };
}