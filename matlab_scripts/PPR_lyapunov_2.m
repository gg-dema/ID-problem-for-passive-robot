
clear all; 
% state variable, original (Q) and after map(Theta)
% syms q1 q2 q3 dq1 dq2 dq3; --> recover from mapping 
syms t1 t2 t3 dt1 dt2 dt3; 

% robot dynamic parameter
m = sym("m", [3, 1]); % mass
d = sym("d", [3, 1]); % CoM_position_respect_joint
I = sym("I", [3, 1]); % Inertia_xx_yy_zz
l = sym("l", [3, 1]); % lenght_link
fv = sym("fv", [3, 1]); % coef viscous friction
fc = sym("fc", [3, 1]); % coef columb friction 

% input : 2D force applied at EE level
F = sym('F', [2, 1]);

a1 = m(1) + m(2) + m(3);
a2 = m(2) + m(3);
a3 = I(3) + m(3) * (d(3)*d(3));
a4 = m(3) * d(3);

%%

% mapping  forward t = h(q) 
%t1 = q2 + l(3)*cos(q3);
%t2 = q1 + l(3)*sin(q3);
%t3 = q3; 

% mapping backward q = h(t) 
q3 = t3; 
q1 = t2 - l(3)*sin(t3);
q2 = t1 - l(3)*cos(t3);



jac_h = [
    0 1, -l(3)*sin(q3);
    1 0 l(3)*cos(q3);
    0 0 1;
];

jac_h_inv = inv(jac_h);

dq = jac_h_inv * [dt1; dt2; dt3];
dq1 = dq(1);
dq2 = dq(2);
dq3 = dq(3);

Actuation = [1, 0; 0, 1; 0, 0];

M_of_q = [ 
    a1 0 0;
    0 a2 (-a4 * sin(q3));
    0 (-a4 * sin(q3)) a3;
]; 
 
C = [
    -a4 * (dq3^2) * sin(q3);
    -a4 * (dq3^2) * cos(q3);
    0;
];

jac_h_dot = [
    0 0 -l(3)*cos(q3)*dq3;
    0 0 -l(3)*sin(q3)*dq3;
    0 0 0;
];

M_of_theta = simplify(jac_h_inv.' * M_of_q * jac_h_inv);
%%

Fv = [ % viscous friction term
    fv(1) 0 0;
    0 fv(2) 0;
    0 0 fv(3);
];

Fc = [ % columb friction term
    fc(1) 0 0;
    0 fc(2) 0;
    0 0 fc(3);
]; 


Friction_term = simplify(Fv*dq + Fc*sign(dq));



%%
theta_dot = simplify(jac_h * dq);
theta_dot_dot = simplify((inv(M_of_theta) * Actuation * F)) + ...
    + simplify((jac_h_dot * jac_h_inv * theta_dot)) - C; % - Friction_term; 

theta_dot_dot = simplify(theta_dot_dot)
%%

% ps: notice the squar:  .^ and not ^ ---> element wise

theta_dot_square = simplify(theta_dot.^2);
theta_dot_dot_square = simplify(theta_dot_dot.^2);


%% 

V = theta_dot_square(1) + theta_dot_square(2) + theta_dot_square(3) + theta_dot_dot_square(1) + theta_dot_dot_square(2) + theta_dot_dot_square(3)
simplify(V)

%%

V = 1/2*(t1*t1 + t2*t2 + theta_dot.' * theta_dot);
x = [t1, t2, t3, dt1, dt2, dt3].' ; 
x_dot = [theta_dot; theta_dot_dot];
V_prime = jacobian(V, x);
% look for matematica 
V_dot = V_prime * x_dot;
