import torch
from aggregate_update_methods import train_and_plot_all_combinations

def main():
    print("Начало эксперимента")
    print("Загрузка графа ethereum_graph.pt...")
    data = torch.load('ethereum_graph.pt', weights_only=False)
    print(f"Граф загружен: {data.num_nodes} вершин, {data.num_edges} рёбер")
    
    print("\nЗапуск обучения всех комбинаций...")
    
    results, histories = train_and_plot_all_combinations(
        data=data,
        epochs=50,        
        save_dir='my_experiment'
    )
    
    print("\nЭксперимент завершён!")
    print("Результаты сохранены в папке etc_experiment_results")

if __name__ == "__main__":
    main()