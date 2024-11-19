import pandas as pd
import torch
import numpy as np

df = pd.read_csv('D:\Sem-5\ML\LAB\LAB3\employeeData.csv')
data = df.to_numpy(dtype=np.float32)
tensor_data = torch.from_numpy(data)
cols = list(df.columns)

def get_entropy_of_dataset(tensor: torch.Tensor):
    total_inst= tensor.size(0)
    count = torch.bincount(tensor[:, -1].long())
    probs = count.float() / total_inst
    
    # entropy calculation
    entropy = -torch.sum(probs * torch.log2(probs))
    return entropy.item()

entropy_value = get_entropy_of_dataset(tensor_data)



# input:tensor,attribute number 
# output:int/float
def get_avg_info_of_attribute(tensor: torch.Tensor, attribute:int):
    """Return avg_info of the attribute provided as parameter"""
    attr_vals = tensor[:, attribute].unique()
    # print("attribute values", attr_vals)
    total_count = tensor.size(0)
    weighted_entropy = 0.0

    for value in attr_vals:
        subset = tensor[tensor[:, attribute] == value]
        subset_entropy = get_entropy_of_dataset(subset)
        weighted_entropy += (subset.size(0) / total_count) * subset_entropy
    
    return weighted_entropy

a = torch.tensor(df.values, dtype=torch.float32)
average = get_avg_info_of_attribute(a,2)

    
    
# output:int/float
def get_information_gain(tensor:torch.Tensor, attribute:int):
    """Return Information Gain of the attribute provided as parameter"""
    initial_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    return initial_entropy - avg_info

a = torch.tensor(df.values, dtype=torch.float32)

ig = get_information_gain(a, 2)



# input: tensor
# output: ({dict},int)
def get_selected_attribute(tensor:torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute

    example : ({0: 0.123, 1: 0.768, 2: 1.23} , 2)
    """
    ig_dict = {}
    
    for i in range(tensor.size(1) - 1): 
        ig = get_information_gain(tensor, i)
        ig_dict[i] = ig
    
    
    selected_attribute = max(ig_dict, key=ig_dict.get)
    
    return ig_dict, selected_attribute

ig_dict, selected_attribute = get_selected_attribute(tensor_data)
