import torch

def update_dataset_and_dataloader(dataset, data_dir, batch_size, workers) -> None:
    dataset.update_data(data_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    return dataset, data_loader