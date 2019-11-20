#include "graph.h"
#include <stdexcept>
#include <cmath>
#include <map>
#include <iostream>
#include "randhelpers.h"

namespace neural_network
{
    double InternalLoss(std::vector<double> expected, std::vector<std::shared_ptr<Variable>> actual)
    {
        double sum = 0;
        for(unsigned long i = 0; i < actual.size(); i++)
        {
            sum += ::pow(expected[i] - actual[i]->Value(), 2);
        }
        sum *= 0.5 * (1 / actual.size());

        return sum;
    }

    Graph::Graph()
    {
        LearningRate(0.05); // Default.
    }

    void Graph::LearningRate(double lr)
    {
        this->learning_rate = lr;
    }

    double Graph::LearningRate()
    {
        return this->learning_rate;
    }

    std::shared_ptr<Variable> Graph::BuildInput()
    {
        std::shared_ptr<Variable> input(new Variable(0));
        this->inputs.push_back(input);
        return input;
    }

    std::shared_ptr<Variable> Graph::BuildLearnable()
    {
        std::shared_ptr<Variable> learnable(new Variable());
        this->learnables.push_back(learnable);
        return learnable;
    }

    // Do I need?
    std::shared_ptr<Variable> Graph::BuildConstant(double value)
    {
        std::shared_ptr<Variable> constant(new Variable(value));
        this->constants.push_back(constant);
        return constant;
    }

    void Graph::AddOutput(std::shared_ptr<Variable> out)
    {
        this->outputs.push_back(out);
    }

    // Expects input to be in order of creation.
    std::vector<double> Graph::Forward(std::vector<double> arg_inputs)
    {
        if(arg_inputs.size() != this->inputs.size())
        {
            throw std::invalid_argument("Input size and expected input size do not match.");
        }

        for(unsigned long i = 0; i < this->inputs.size(); i++)
        {
            this->inputs[i]->Value(arg_inputs[i]);
        }

        std::vector<double> ret_vector;
        for(auto j : this->outputs)
        {
            ret_vector.push_back(j->Value());
        }

        return ret_vector;
    }

    std::vector<double> Graph::Train(
        std::vector<std::vector<double>> arg_inputs, 
        std::vector<std::vector<double>> labels,
        int batch_size,
        int epochs)
    {
        if(arg_inputs.size() == 0)
        {
            throw std::invalid_argument("Input given with no rows.");
        }

        if(arg_inputs[0].size() != this->inputs.size())
        {
            throw std::invalid_argument("Input size and expected input size do not match.");
        }

        if(labels.size() == 0)
        {
            throw std::invalid_argument("Labels given with no rows.");
        }

        if(labels[0].size() != this->outputs.size())
        {
            throw std::invalid_argument("Label size and output size do not match.");
        }

        if(arg_inputs.size() != labels.size())
        {
            throw std::invalid_argument("Input and label size do not match.");
        }

        std::vector<double> losses;
        std::map<std::shared_ptr<Variable>,double> cached_updates;

        for(int i = 0; i < epochs; i++)
        {
            for(auto j : this->learnables)
            {
                cached_updates[j] = 0.0;
            }

            for(int j = 0; j < 1; j++)
            {
                int sample = randMinToMax(0, inputs.size() + 1);

                auto X_sample = arg_inputs[sample];
                auto label = labels[sample];

                Forward(X_sample);

                losses.push_back(InternalLoss(label, this->outputs));

                for(unsigned long k = 0; k < this->outputs.size(); k++)
                {
                    auto current_output = this->outputs[k];

                    current_output->Gradient(1.0);
                    double error_derivative = -(label[k] - current_output->Value());

                    for(auto j : this->learnables)
                    {
                        cached_updates[j] += error_derivative * j->Gradient();
                    }

                    current_output->Gradient(0.0);
                }
            }

            for(auto j : this->learnables)
            {
                j->Value(j->Value() - this->learning_rate * cached_updates[j]);
            }

            Reset();

            /*std::cout
                << "Epoch: " << i
                << " Loss: " << losses.back()
                << std::endl;*/
        }

        return losses;
    }

    void Graph::Backward(std::vector<double> expected)
    {
        if(expected.size() != this->outputs.size())
        {
            throw std::invalid_argument("Expected size and output size do not match.");
        }

        for(unsigned long i = 0; i < this->outputs.size(); i++)
        {
            auto curr_output = this->outputs[i];

            curr_output->Gradient(1.0);
            double error_derivative = -(expected[i] - curr_output->Value());

            double lr_error_product = error_derivative * this->learning_rate;
            for(auto j : this->learnables)
            {
                j->Value(j->Value() - lr_error_product * j->Gradient());
            }

            curr_output->Gradient(0.0);
        }
    }

    void Graph::Reset()
    {
        for(auto i : this->inputs)
        {
            i->ClearGradient();
        }

        for(auto i : this->learnables)
        {
            i->ClearGradient();
        }

        for(auto i : this->constants)
        {
            i->ClearGradient();
        }

        for(auto i : this->outputs)
        {
            i->ClearGradient();
        }
    }

    double Graph::Loss(std::vector<double> expected)
    {
        if(expected.size() != this->outputs.size())
        {
            throw std::invalid_argument("Expected size and output size do not match.");
        }

        return InternalLoss(expected, this->outputs);
    }
}