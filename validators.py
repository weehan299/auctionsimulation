
    
def validate_action_space_num(self,attribute,value):
    if value <= 0:
        raise ValueError("number of elements in action space should be positive")

def validate_total_periods(self,attribute,value):
    if value <= 0:
        raise ValueError("total number of periods in action space should be positive")

def validate_gamma(self, attribute, value):
    if value > 1 or value < 0:
        raise ValueError("gamma shld be between 0 and 1")

def validate_learning_rate(self, attribute, value):
    if value > 1 or value < 0:
        raise ValueError("learning rate shld be between 0 and 1")

def validate_alpha(self, attribute, value):
    if value < 1 or value > 2:
        raise ValueError("alpha shld be between 1 and 2")

def validate_beta(self, attribute, value):
    if value < 0:
        raise ValueError("beta shld be positive")

def validate_epsilon(self, attribute, value):
    if value < 0:
        raise ValueError("epsilon shld be positive")