import numpy as np
input_data = np.array([2,3])
weights={'node_0':np.array([1,1]),
         'node_1':np.array([-1,-1]),
         'output':np.array([2,-1])
         }
def relu(input):
    output = max(input, 0)
    return(output)
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)
hidden_layer_outputs = np.array([node_0_output, node_1_output])
print(f"The hidden layer values are ",hidden_layer_outputs)
model_output = (hidden_layer_outputs * weights['output']).sum()
print(f"The final output is",model_output)
input("Press any key to continue...")