import torch
from aggregate_update_methods import train_and_plot_all_combinations

data = torch.load('ethereum_graph.pt')

results, histories = train_and_plot_all_combinations(
    data=data,
    epochs=50,  
    save_dir='my_experiment' 
)