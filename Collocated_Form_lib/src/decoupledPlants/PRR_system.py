

class PRR():

    def __init__(self) -> None:
        
        def H_mapping():
            print("init h mapping")

        def Q_robot():
            print("init q rob")

        def wrap_subsystem(plant, mapping):
            pass

        self.Plant = Q_robot()
        self.Mapping = H_mapping()

        wrap_subsystem(self.Plant, self.Mapping)


         

    
