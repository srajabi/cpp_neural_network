#include <iostream>
#include "dual.h"
#include "variable.h"
#include <iomanip>

using namespace neural_network;


int main(int argc, char** argv)
{
    std::cout << "hello world" << std::endl;

/* DUAL NUMBERS
    cpp_nn::Dual x(5, 1);
    cpp_nn::Dual y(6, 0);

    auto w1 = pow(x, 2);

    cpp_nn::Dual f = y * w1;
 
    std::cout << "f: " << f.getDerivative() << " "
              << "x: " << x.getDerivative() << " "
              << "w1: " << w1.getDerivative() << " " 
              << "y: " << y.getDerivative() << " "
              << std::endl;
*/


/* REVERSE-MODE AD *//*
    std::shared_ptr<Variable> x(new Variable(0.5));
    std::shared_ptr<Variable> y(new Variable(4.2));

    auto a = sin(x);
    auto b = x * y;
    auto c = a + b;

    std::cout << c->getValue() << std::endl;

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;

    auto z = c;

    std::cout << z << std::endl;
    std::cout << x << std::endl;
    std::cout << y << std::endl;

    z->setGradient(1);
    std::cout << x->getGradient() << std::endl;
    std::cout << y->getGradient() << std::endl;

    std::cout << z << std::endl;
    std::cout << x << std::endl;
    std::cout << y << std::endl;

    return 0;*/

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
        // PRINT THE BITCH
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
        x1->setValue(arg_x1);
        x2->setValue(arg_x2);

        return output->getValue();
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

    output->setGradient(1.0);
    x1->getGradient();
    x2->getGradient();
    b0->getGradient();
    b1->getGradient();
    w00->getGradient();
    w01->getGradient();
    w02->getGradient();
    w03->getGradient();
    w04->getGradient();
    w05->getGradient();
    w10->getGradient();
    w11->getGradient();
    w12->getGradient();
    dbg_print();

    double alpha = 0.05; // learning rate

    auto backward_pass = [=](double expected_label, std::shared_ptr<Variable> output_label)
    {
        double difference = expected_label - output_label->getValue();

        // J(), the loss function
        double loss = 0.5 * ::pow(difference, 2);

        // dJ/dpredicted = dJ/dy, where predicted = output_label
        double error_derivative = -difference;

        // seed dJ/dJ = 1
        output->setGradient(1.0);
        x1->getGradient();
        x2->getGradient();

        b0->getGradient();
        b1->getGradient();

        w00->getGradient();
        w01->getGradient();

/*
        std::cout << " ** dypred/w01 " << w01->getGradient() << std::endl;
        std::cout << " ** dL/dw01 " << error_derivative * w01->getGradient() << std::endl;
        double dE_dypred = error_derivative;
        double dypred_da2 = w12->getValue() * (output->getValue() * (1 - output->getValue()));
        double da2_dw01 = b1->getValue() * (a2_activated->getValue() * (1 - a2_activated->getValue()));

        std::cout << " ** dypred/w01 " << dypred_da2 * da2_dw01 << std::endl;
        std::cout << " ** dL/dw01 " << dE_dypred * dypred_da2 * da2_dw01 << std::endl;
*/

        w02->getGradient();
        w03->getGradient();
        w04->getGradient();
        w05->getGradient();
        w10->getGradient();
        w11->getGradient();
        w12->getGradient();

        //std::cout << "******** updates ********" << std::endl;

        // dJ/dy * dy/dW = dJ/dW
        w00->setValue(w00->getValue() - error_derivative * alpha * w00->getGradient());
        w01->setValue(w01->getValue() - error_derivative * alpha * w01->getGradient());
        w02->setValue(w02->getValue() - error_derivative * alpha * w02->getGradient());
        w03->setValue(w03->getValue() - error_derivative * alpha * w03->getGradient());
        w04->setValue(w04->getValue() - error_derivative * alpha * w04->getGradient());
        w05->setValue(w05->getValue() - error_derivative * alpha * w05->getGradient());

        w10->setValue(w10->getValue() - error_derivative * alpha * w10->getGradient());
        w11->setValue(w11->getValue() - error_derivative * alpha * w11->getGradient());
        w12->setValue(w12->getValue() - error_derivative * alpha * w12->getGradient());
        
        return loss;
    };

    dbg_print();

    for(int i = 0; i < 100000; i++)
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

/*
    auto linear_function = [](double x0_arg, double x1_arg)
    {
        //return 100;
        return x0_arg;
        //return 0.1 * x0_arg + 3;
    };

    for(int i = 0; i < 100000000; i++)
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
    output->setGradient(1.0);
    x1->getGradient();
    x2->getGradient();

    std::cout << y_fd << std::endl;
    std::cout << x1 << std::endl;
*/
    return 0;
}