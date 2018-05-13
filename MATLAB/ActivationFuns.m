x = (-2:0.1:2);

sigmoid = 1./(1+exp(-x));
relu = max(zeros(size(x)),x);
tanh = (exp(x) - exp(-x))./(exp(x) + exp(-x)); 

figure
subplot(1,3,1);
plot(x, sigmoid, 'LineWidth', 2);
ylim([-2,2])
title('Logistic (Sigmoid)', 'FontSize', 12);
xlabel('x');
ylabel('\sigma (x)');

subplot(1,3,2);
plot(x, relu, 'LineWidth', 2);
ylim([-2,2])
title('Rectified Linear Unit (RELU)', 'FontSize', 12);
xlabel('x');
ylabel('\sigma (x)');

subplot(1,3,3);
plot(x, tanh, 'LineWidth', 2);
ylim([-2,2])
title('Hyperbolic Tangent (tanh)', 'FontSize', 12);
xlabel('x');
ylabel('\sigma (x)');
