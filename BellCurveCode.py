import numpy as np

table_size = 101 # How many samples are there in the look up table, the higher the number the more accurate it is but more memory.
width_of_network = 0.5 # This is a percentage of how much does the bell curve cover, so 1 would be 100%, 0.5 would be 50%, 2 would be 200%, etc.
pos_of_network = 0 # This is where on the network the bell curve starts as a percentage, so 0 is at the start and 1 is 100% of the way through the network.

minimum_value = 0 # When sampling the curve, what is the lowest value, a sort of bias term so it is never 0, if need be.
maximum_value = 5  # When sampling the curve, what is the value at the top of the curve.

movement_per_batch = 0.05 # How much the bell curve moves every batch, this is a percentage of the network, once it has reached the end it will go back to the start of the network

standard_deviation = 0.2
spread = 2 * standard_deviation ** 2

lookup_table = []

def calculate_lookup_table():
    x_range = np.linspace(-0.5, 0.5, table_size)
    return np.exp(-(x_range**2) / spread)

def bell_curve_sample_lookup(x):
    idx = int((x + 0.5) * (table_size - 1))
    idx = max(0, min(idx, table_size - 1))
    y = lookup_table[idx]
    return minimum_value + y * (maximum_value - minimum_value)

#I wanted to see the difference if creating a look up table and performing the 
# calculation for more precition has a big impact, I think the lost precision 
# does not matter as it is so much faster to get a pre-computed matrix
def bellcurve_sample_no_lookup(x):
    y = np.exp(-(x**2) / (2 * spread**2))
    return minimum_value + y * (maximum_value - minimum_value)

lookup_table = calculate_lookup_table()

# After the network has been made you make a matrix with all the parameters 
# but this is not a copy it is direct access to the layers and their derivatives which 
# makes it really easy to cycle over them in the training loop.
params_for_bellcurve = []
for name, param in feedforward_model.named_parameters():
        if param.requires_grad:
            params_for_bellcurve.append(param)
            
# In the train loop
for i, layer in enumerate(params_for_bellcurve):
            # print(layer.grad)
            # print(layer.shape)
            
            # This is probably the most complicated bit, here you calculate the 
            # percentace of how far gone it is through the network with i / (len(params_for_bellcurve) - 1) 
            # Then this gives you a number between 0-1 being 0% through the network or 100%, 
            # then you can minus the pos_of_network 
            # and this is a number between 0 and 1 where it tracks how far through 
            # the bell curve is through the network, the closer this value is to 0, the more in the middle of the bell curve it is
            # meaning that the higher the value is as the peak of the bell curve is in the middle of the matrix in the look up table or the raw computation 
            # you +0.5 the input so that this is the case, 
            # after this you divide it by the width of the network, you can think of this as scaling the bell curve as if you divide it by two it gets closer to the middle.
            multiplier = bell_curve_sample_lookup((i / (len(params_for_bellcurve) - 1) - pos_of_network) / width_of_network)
            layer.grad.data.mul_(multiplier)
            # print(layer.grad)
            # print(multiplier)
            # input()
        
        pos_of_network += movement_per_batch
        if pos_of_network >= 1:
            pos_of_network = 0