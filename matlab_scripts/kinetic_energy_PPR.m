clear all;
pkg load symbolic
 
syms q1 q2 q3 dq1 dq2 dq3;
syms dc1 dc2 dc3;
syms Iz3;
syms m1 m2 m3; 


%%
%compute kin energy 
CoM1 = [0; q1 - dc1; 0];
CoM2 = [q2 - dc2; q1; 0];
CoM3 = [q2 + dc3*cos(q3); 
        q1 + dc3*sin(q3);
        0];

Vc1 = [0, dq1, 0].';
Vc2 = [dq2, dq1, 0].'; 
Vc3 = [dq2 - dc3*sin(q3)*dq3;
        dq1 + dc3*cos(q3)*dq3; 
        0];

T1 = (1/2) * m1 * (Vc1.' * Vc1);
T2 = (1/2) * m2 * (Vc2.' * Vc2);
T3 = simplify( ((1/2) * m3 * (Vc3.' * Vc3)) + ((1/2) * dq3^2 * Iz3) );

T = simplify(expand(T1 + T2 + T3))

%%
%V = 0 at equilibrium ? eq: any combination of (q, 0) 
subs(T, [dq1, dq2, dq3], [0, 0, 0]); % check if at eq V = 0 
collect(T, [dq1, dq2, dq3])

% second condition for T = V ---> V(x) > 0 everywhere:  not my case
