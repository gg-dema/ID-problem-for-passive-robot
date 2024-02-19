clear all; 

% if u wanna run this in octave, add the next line: 
%pkg load symbolic
syms q1 q2 q3 l2 l3; 

% mapping: h(q) = theta
% passive output = position end-effector (x,y)
%

h_of_theta = [...
    q1 + l2*cos(q2) + l3*cos(q2+q3);
    l2*sin(q2) + l3*sin(q2+q3);
    q3;
    ];

jac_h = jacobian(h_of_theta, [q1, q2, q3])

%% Actuation matrix of the system: 

% the robot is driven by a external force that act on the end-effector, 
% so the actuation matrix correspond to the jacobian of the direct
% kinematics

kinematics = [  q1 + l2*cos(q2) + l3*cos(q2+q3);
                l2*sin(q2) + l3*sin(q2+q3);
                ];

jac_ee = jacobian(kinematics, [q1, q2, q3])
ActuationMatrix = jac_ee.'

%% Solution of ID problem:

% solution : DynModel = { J_h ^-T * ActuationMatrix } * F = [F1, F2, 0]
simplify(inv(jac_h.') * ActuationMatrix)