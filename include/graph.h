#pragma once

#include <vector>
#include <ostream>
#include "variable.h"

namespace neural_network
{
    class Graph
    {
        public:
            Graph();

            std::shared_ptr<Variable> BuildInput();
            std::shared_ptr<Variable> BuildLearnable();
            std::shared_ptr<Variable> BuildConstant(double value);

            void AddOutput(std::shared_ptr<Variable> out);

            // TODO Factor out learning parts as it violates SRP.
            std::vector<double> Forward(std::vector<double> inputs);
            void Backward(std::vector<double> expected);

            std::vector<double> Train(
                std::vector<std::vector<double>> inputs, 
                std::vector<std::vector<double>> labels, 
                int batch_size, 
                int epochs);

            void Reset();

            void LearningRate(double lr);
            double LearningRate();

            double Loss(std::vector<double> expected);

        private:
            std::vector<std::shared_ptr<Variable>> inputs;
            std::vector<std::shared_ptr<Variable>> learnables;
            std::vector<std::shared_ptr<Variable>> constants;
            std::vector<std::shared_ptr<Variable>> outputs;
            double learning_rate;
    };
}