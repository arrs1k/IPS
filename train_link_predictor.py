import torch
import argparse
from torch_geometric.transforms import RandomLinkSplit
from ethereum_link_predictor import EthereumLinkPredictor, EthereumLinkPredictionTrainer


def main():
    parser = argparse.ArgumentParser(description="Обучение модели link prediction на графе Ethereum")
    parser.add_argument("--graph_path", default="ethereum_graph.pt", help="Путь к файлу с графом")
    parser.add_argument("--save_model", default="ethereum_link_model.pt", help="Путь для сохранения модели")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох обучения")
    parser.add_argument("--batch_size", type=int, default=1024, help="Размер батча")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Размер скрытого слоя")
    parser.add_argument("--lr", type=float, default=0.001, help="Скорость обучения")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    args = parser.parse_args()

    # 1. Загрузка графа
    print("=" * 60)
    print("ЗАГРУЗКА ГРАФА")
    print("=" * 60)
    data = torch.load(args.graph_path)
    print(f"Граф загружен из {args.graph_path}")
    print(f"  - Вершин: {data.num_nodes:,}")
    print(f"  - Рёбер: {data.num_edges:,}")
    print(f"  - Размерность признаков: {data.x.shape[1]}")

    # 2. Разделение данных на train/val/test
    print("\n" + "=" * 60)
    print("РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 60)
    transform = RandomLinkSplit(
        is_undirected=False,
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = transform(data)
    print(f"Train: {len(train_data)} графов, {train_data.edge_label_index.shape[1]} пар рёбер")
    print(f"Val: {val_data.edge_label_index.shape[1]} пар рёбер")
    print(f"Test: {test_data.edge_label_index.shape[1]} пар рёбер")

    # 3. Настройка устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    print(f"\nУстройство: {device}")

    # 4. Создание модели
    print("\n" + "=" * 60)
    print("СОЗДАНИЕ МОДЕЛИ")
    print("=" * 60)
    model = EthereumLinkPredictor(
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden_dim,
        out_channels=64,
        num_layers=2,
        dropout=args.dropout
    )
    print(f"Модель: {model.__class__.__name__}")
    print(f"  - Входные каналы: {data.x.shape[1]}")
    print(f"  - Скрытые каналы: {args.hidden_dim}")
    print(f"  - Dropout: {args.dropout}")
    print(f"  - Параметров: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Оптимизатор и функция потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 6. Создание трейнера
    trainer = EthereumLinkPredictionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_neighbors=[10, 5],
        batch_size=args.batch_size
    )

    # 7. Обучение
    print("\n" + "=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)
    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        verbose=True,
        eval_metrics=True
    )

    # 8. Оценка на тестовых данных
    print("\n" + "=" * 60)
    print("ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ")
    print("=" * 60)
    test_auc = trainer.evaluate_auc(test_data)
    test_loss = trainer.evaluate_loss(test_data)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # 9. Сохранение модели
    trainer.save_model(args.save_model)

    # 10. Вывод итогов
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Лучшая Val AUC: {max(history['val_auc'] if history['val_auc'] else [0]):.4f}")
    print(f"Лучшая Val Loss: {min(history['val_loss'] if history['val_loss'] else [0]):.4f}")
    print(f"Модель сохранена в {args.save_model}")

    # 11. Пример предсказания
    print("\n" + "=" * 60)
    print("ПРИМЕР ПРЕДСКАЗАНИЯ")
    print("=" * 60)
    predictions = trainer.predict(test_data, test_data.edge_label_index[:, :10])
    print(f"Предсказания для первых 10 тестовых пар:")
    for i, pred in enumerate(predictions):
        print(f"  Пара {i + 1}: вероятность связи = {pred:.4f}")


if __name__ == "__main__":
    main()