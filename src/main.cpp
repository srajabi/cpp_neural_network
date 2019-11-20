#include <iostream>
#include "dual.h"
#include "variable.h"
#include <iomanip>

using namespace neural_network;

int main()
{
    std::srand(1);

    // Input Layer
    std::shared_ptr<Variable> b0(new Variable(1)); // bias
    std::shared_ptr<Variable> x1(new Variable(0)); // input x1
    std::shared_ptr<Variable> x2(new Variable(0)); // input x2

    // weight, randomly initialized
    std::shared_ptr<Variable> w00(new Variable());
    std::shared_ptr<Variable> w01(new Variable());
    std::shared_ptr<Variable> w02(new Variable());
    std::shared_ptr<Variable> w03(new Variable());
    std::shared_ptr<Variable> w04(new Variable());
    std::shared_ptr<Variable> w05(new Variable());

    // Hidden Layer 
    std::shared_ptr<Variable> b1(new Variable(1)); // bias

    // weights
    std::shared_ptr<Variable> w10(new Variable());
    std::shared_ptr<Variable> w11(new Variable());
    std::shared_ptr<Variable> w12(new Variable());

    // intermediary steps
    std::shared_ptr<Variable> a1_activity;
    std::shared_ptr<Variable> a1_activated;
    std::shared_ptr<Variable> a2_activity;
    std::shared_ptr<Variable> a2_activated;
    std::shared_ptr<Variable> output_activity;
    std::shared_ptr<Variable> output;

    auto build_graph = [&]()
    {
        a1_activity = x1 * w02 + x2 * w04 + b0 * w00;
        a1_activated = sigmoid(a1_activity);

        a2_activity = x1 * w03 + x2 * w05 + b0 * w01;
        a2_activated = sigmoid(a2_activity);

        output = a1_activated * w11 + a2_activated * w12 + b1 * w10;
        //output = sigmoid(output_activity);

        return output;
    };

    auto dbg_print = [&]()
    {
        std::cout << " *** dbg *** " << std::endl;

        std::cout << "b0 : " << b0 << std::endl;
        std::cout << "x1 : " << x1 << std::endl;
        std::cout << "x2 : " << x2 << std::endl;
        std::cout << "w00 : " << w00 << std::endl;
        std::cout << "w01 : " << w01 << std::endl;
        std::cout << "w02 : " << w02 << std::endl;
        std::cout << "w03 : " << w03 << std::endl;
        std::cout << "w04 : " << w04 << std::endl;
        std::cout << "w05 : " << w05 << std::endl;
        std::cout << "b1 : " << b1 << std::endl;
        std::cout << "w10 : " << w10 << std::endl;
        std::cout << "w11 : " << w11 << std::endl;
        std::cout << "w12 : " << w12 << std::endl;
        std::cout << "a1_activity : " << a1_activity << std::endl;
        std::cout << "a1_activated: " << a1_activated << std::endl;
        std::cout << "a2_activity: " << a2_activity << std::endl;
        std::cout << "a2_activated: " << a2_activated << std::endl;
        //std::cout << "output_activity: " << output_activity << std::endl;
        std::cout << "output: " << output << std::endl;
    };

    // Output
    output = build_graph();

    auto forward_pass = [=](double arg_x1, double arg_x2)
    {
        x1->Value(arg_x1);
        x2->Value(arg_x2);

        return output->Value();
    };

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
        std::cout << "nn(x1, x2): " << forward_pass(input_x1, input_x2) << std::endl;
    }

    output->Gradient(1.0);
    x1->Gradient();
    x2->Gradient();
    b0->Gradient();
    b1->Gradient();
    w00->Gradient();
    w01->Gradient();
    w02->Gradient();
    w03->Gradient();
    w04->Gradient();
    w05->Gradient();
    w10->Gradient();
    w11->Gradient();
    w12->Gradient();
    dbg_print();

    double alpha = 0.05; // learning rate

    auto backward_pass = [=](double expected_label, std::shared_ptr<Variable> output_label)
    {
        double difference = expected_label - output_label->Value();

        // J(), the loss function
        double loss = 0.5 * ::pow(difference, 2);

        // dJ/dpredicted = dJ/dy, where predicted = output_label
        double error_derivative = -difference;

        // seed dJ/dJ = 1
        output->Gradient(1.0);
        x1->Gradient();
        x2->Gradient();

        b0->Gradient();
        b1->Gradient();

        w00->Gradient();
        w01->Gradient();

/*
        std::cout << " ** dypred/w01 " << w01->Gradient() << std::endl;
        std::cout << " ** dL/dw01 " << error_derivative * w01->Gradient() << std::endl;
        double dE_dypred = error_derivative;
        double dypred_da2 = w12->Value() * (output->Value() * (1 - output->Value()));
        double da2_dw01 = b1->Value() * (a2_activated->Value() * (1 - a2_activated->Value()));

        std::cout << " ** dypred/w01 " << dypred_da2 * da2_dw01 << std::endl;
        std::cout << " ** dL/dw01 " << dE_dypred * dypred_da2 * da2_dw01 << std::endl;
*/

        w02->Gradient();
        w03->Gradient();
        w04->Gradient();
        w05->Gradient();
        w10->Gradient();
        w11->Gradient();
        w12->Gradient();

        //std::cout << "******** updates ********" << std::endl;

        // dJ/dy * dy/dW = dJ/dW
        w00->Value(w00->Value() - error_derivative * alpha * w00->Gradient());
        w01->Value(w01->Value() - error_derivative * alpha * w01->Gradient());
        w02->Value(w02->Value() - error_derivative * alpha * w02->Gradient());
        w03->Value(w03->Value() - error_derivative * alpha * w03->Gradient());
        w04->Value(w04->Value() - error_derivative * alpha * w04->Gradient());
        w05->Value(w05->Value() - error_derivative * alpha * w05->Gradient());

        w10->Value(w10->Value() - error_derivative * alpha * w10->Gradient());
        w11->Value(w11->Value() - error_derivative * alpha * w11->Gradient());
        w12->Value(w12->Value() - error_derivative * alpha * w12->Gradient());
        
        return loss;
    };

    dbg_print();

    for(int i = 0; i < 10000; i++)
    {
        double x1_arg = test_data[i%test_data.size()].first;
        double x2_arg = test_data[i%test_data.size()].second;

        double y = xor_function(x1_arg, x2_arg);

        double y_hat = forward_pass(x1_arg, x2_arg);

        //std::cout << "******** bw pass ********" << std::endl;
        double loss = backward_pass(y, output);

        std::cout << std::setprecision(15) << "epoch: " << i << " x1: " << x1 << " x2: " << x2 << " y: " << y << " yh: " << y_hat << " loss: " << loss << std::endl;

        x1->ClearGradient();
        x2->ClearGradient();
        b0->ClearGradient();
        b1->ClearGradient();
        w00->ClearGradient();
        w01->ClearGradient();
        w02->ClearGradient();
        w03->ClearGradient();
        w04->ClearGradient();
        w05->ClearGradient();
        w10->ClearGradient();
        w11->ClearGradient();
        w12->ClearGradient();
    }

    dbg_print();

    auto linear_function = [](double x0_arg, double x1_arg)
    {
        //return 100;
        return x0_arg;
        //return 0.1 * x0_arg + 3;
    };

    for(int i = 0; i < 10000; i++)
    {
        double x1_arg = i%80 - 40;
        double x2_arg = 0;

        double y = linear_function(x1_arg, x2_arg);

        double y_hat = forward_pass(x1_arg, x2_arg);

        //std::cout << "******** bw pass ********" << std::endl;
        double loss = backward_pass(y, output);

        std::cout << std::setprecision(15) << "epoch: " << i << " x1: " << x1 << " x2: " << x2 << " y: " << y << " yh: " << y_hat << " loss: " << loss << std::endl;

        x1->ClearGradient();
        x2->ClearGradient();
        b0->ClearGradient();
        b1->ClearGradient();
        w00->ClearGradient();
        w01->ClearGradient();
        w02->ClearGradient();
        w03->ClearGradient();
        w04->ClearGradient();
        w05->ClearGradient();
        w10->ClearGradient();
        w11->ClearGradient();
        w12->ClearGradient();
    }

    double y_fd = (forward_pass(1 + 0.00001, 1) - forward_pass(1, 1)) / 0.00001;
    output->Gradient(1.0);
    x1->Gradient();
    x2->Gradient();

    std::cout << y_fd << std::endl;
    std::cout << x1 << std::endl;
    return EXIT_SUCCESS;
}