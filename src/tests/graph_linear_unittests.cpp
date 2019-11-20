#include "graph.h"
#include "gtest/gtest.h"

using namespace neural_network;
namespace
{
    TEST(Graph, LinearFunction)
    {
        std::srand(1);

        Graph graph;
        auto b0 = graph.BuildConstant(1);
        auto x0 = graph.BuildInput();

        auto w00 = graph.BuildLearnable();
        auto w01 = graph.BuildLearnable();
        auto w02 = graph.BuildLearnable();
        auto w03 = graph.BuildLearnable();

        auto b1 = graph.BuildConstant(1);

        auto w10 = graph.BuildLearnable();
        auto w11 = graph.BuildLearnable();
        auto w12 = graph.BuildLearnable();

        auto a1_activity = x0 * w02 + b0 * w00;
        auto a2_activity = x0 * w03 + b0 * w01;

        auto output = a1_activity * w11 + a2_activity * w12 + b1 * w10;

        graph.AddOutput(output);
        graph.LearningRate(0.05);

        auto linear_function = [](double input_0)
        {
            return input_0 * 0.5 + 7;
        };

        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> labels;
        for(int i = 0; i < 100; i++)
        {
            double in = i;

            inputs.push_back({in});

            double y = linear_function(in);

            labels.push_back({y});
        }

        graph.Train(inputs, labels, 100, 100000);

        for(int i = 0; i < 100; i++)
        {
            double in = i;

            auto y_pred = graph.Forward({in});
            auto y_actual = linear_function(in);

            std::cout << "Test: "
                << " in: " << in
                << " y_pred: " << y_pred[0]
                << " y_actual: " << y_actual
                << std::endl;

            EXPECT_NEAR(y_pred[0], y_actual, 0.01);
        }
    }

    TEST(Graph, XorFunction)
    {
        std::srand(1);

        Graph graph;
        auto b0 = graph.BuildConstant(1);
        auto x0 = graph.BuildInput();
        auto x1 = graph.BuildInput();

        auto w00 = graph.BuildLearnable();
        auto w01 = graph.BuildLearnable();
        auto w02 = graph.BuildLearnable();
        auto w03 = graph.BuildLearnable();
        auto w04 = graph.BuildLearnable();
        auto w05 = graph.BuildLearnable();

        auto b1 = graph.BuildConstant(1);

        auto w10 = graph.BuildLearnable();
        auto w11 = graph.BuildLearnable();
        auto w12 = graph.BuildLearnable();

        auto a1_activity = x0 * w02 + x1 * w04 + b0 * w00;
        auto a1_activated = sigmoid(a1_activity);

        auto a2_activity = x0 * w03 + x1 * w05 + b0 * w01;
        auto a2_activated = sigmoid(a2_activity);

        auto output = a1_activated * w11 + a2_activated * w12 + b1 * w10;

        graph.AddOutput(output);
        graph.LearningRate(0.05);

        auto xor_function = [](double x0, double x1)
        {
            bool b0 = x0 == 0.0;
            bool b1 = x1 == 0.0;

            bool y = b0 ^ b1;

            return y ? 1.0 : 0.0;
        };

        const std::vector<std::pair<double, double>> test_data = 
        {
            {0, 0}, // = 0
            {0, 1}, // = 1
            {1, 1}, // = 0
            {1, 0}  // = 1
        };

        for(auto input : test_data)
        {
            double input_x1 = input.first;
            double input_x2 = input.second;

            std::cout << "xor(x1, x2): " << xor_function(input_x1, input_x2) << std::endl;
            std::cout << "nn(x1, x2): " << graph.Forward({input_x1, input_x2})[0] << std::endl;
        }

        for(int i = 0; i < 0; i++)
        {
            double x1_arg = test_data[i%test_data.size()].first;
            double x2_arg = test_data[i%test_data.size()].second;

            double y = xor_function(x1_arg, x2_arg);
            double y_hat = graph.Forward({x1_arg, x2_arg})[0];

            graph.Backward({y});

            std::cout <<
                "Epoch: " << i <<
                " x1: " << x1_arg <<
                " x2: " << x2_arg <<
                " y: " << y <<
                " y_hat: " << y_hat << std::endl;

            graph.Reset();
        }

        std::vector<std::vector<double>> inputs = {
            {0, 0},
            {0, 1},
            {1, 1},
            {1, 0}
        };

        std::vector<std::vector<double>> labels = {
            {0},
            {1},
            {0},
            {1}
        };

        graph.Train(inputs, labels, 1, 100000);

        for(auto input : test_data)
        {
            double input_x1 = input.first;
            double input_x2 = input.second;

            std::cout << "xor(x1, x2): " << xor_function(input_x1, input_x2) << std::endl;
            std::cout << "nn(x1, x2): " << graph.Forward({input_x1, input_x2})[0] << std::endl;

            EXPECT_NEAR(graph.Forward({input_x1, input_x2})[0], xor_function(input_x1, input_x2), 0.001);
        }
    }
}