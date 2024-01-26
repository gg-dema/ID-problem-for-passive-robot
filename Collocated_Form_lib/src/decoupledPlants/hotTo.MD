Hot to use and extend this class:

        For each robot, define an interface in the decoupled-Plants dir.
        This robot will be a py file that connect two module:
            - Plan-[robot_model] 
            - mapping-system-[robot-model]
        This module should load also the parameters form some urdf/sdf file, 
        so, each robot interace should read the correct sdf file for the folder:
            - ../models/[urdf/sdf]
        

Ps: 
    for now i'll desing a time delay system for extract the correct info from the state for the controller.
    This system will work only with spatial force applied at the end-effector level. 

    Later on, a new module will be integrated and will contains the different option for create a input for the controller
    