f = @(x) min(max(-1, 4*(x - 0.2)), 1);

x = linspace(-1, 1, 100);
y = arrayfun(f, x);
xi = linspace(-1, 1, 1000);
yi = interp1(x, y, xi, 'spline');

figure;
plot(x, y, 'o', xi, yi, '-');
legend('Sample Points', 'Spline Interpolation');
title('Spline Interpolation');
xlabel('x');
ylabel('f(x)');
print('spline_interpolation', '-dpng');

A = 1;
beta = 0.99;
delta = 0.025;
alpha = 0.36;
kmin = 0.06;
kmax = 12;
tol = 0.01;
m = 300;

k = kmin:kmin:kmax;

[V, policy] = run_bellman(k, A, beta, delta, alpha, tol, m);
[V_trivial, policy_trivial] = run_bellman(k, A, beta, 1, alpha, tol, m);

figure;
plot(k, V);
title('Value Function');
xlabel('Capital Stock k');
ylabel('Value Function V(k)');
print('value_function', '-dpng');

figure;
plot(k, policy, k, policy_trivial);
title('Policy Function');
xlabel('Capital Stock k');
ylabel('Optimal k''');
legend('delta = 0.025', 'delta = 1');
print('policy_function', '-dpng');

function [V, policy] = run_bellman(k, A, beta, delta, alpha, tol, m)
n = length(k);
V = zeros(1, n);

for iter = 1:m
    Vnew = zeros(1, n);
    for i = 1:n
        obj = @(j) (log(A*k(i)^alpha + (1-delta)*k(i) - k(j)) + beta*V(j));
        
        jmax = find(k <= A*k(i)^alpha + (1-delta)*k(i), 1, 'last');
        [val, ip] = max(arrayfun(obj, 1:jmax));
        
        if A*k(i)^alpha + (1-delta)*k(i) - k(ip) <= 0
            val = -Inf;
        end
        
        Vnew(i) = val;
    end
    
    if max(abs(V - Vnew)) < tol
        break;
    end
    V = Vnew;
end

policy = zeros(1, length(k));
for i = 1:n
    obj = @(kp) -(log(A*k(i)^alpha + (1-delta)*k(i) - kp) + beta*interp1(k, V, kp, 'spline'));
    [kp_opt, ~] = fminbnd(obj, 0, A*k(i)^alpha + (1-delta)*k(i));
    policy(i) = kp_opt;
end
end
