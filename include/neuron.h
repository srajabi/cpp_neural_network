#pragma once

#include <functional>
#include <vector>

namespace cpp_nn
{
    class Neuron 
    {
        public:
            void set_function(std::function<float(float)> activation_function);
            float ativate(float in);

            float get_weight();
            void set_weight(float weight);

        private:
            std::function<float(float)> activation;
            float weight;
            std::vector<Neuron> outbound_connections;
    };
}