import hydra
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hydra.utils import instantiate


from utils import init_run, get_class_weights
from preprocessing import build_vocabulary


@hydra.main(config_path="configs", version_base=None)
def main(config):
    device = init_run(config)

    model = instantiate(config=config.model).to(device)

    if config.general.compute_class_weights:
        class_weights = get_class_weights(config.dataset.path, config.dataset.train)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = CrossEntropyLoss(weight=class_weights, reduction="mean")
    else:
        criterion = CrossEntropyLoss()

    optimizer = Adam(lr=config.optimizer.learning_rate, params=model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    vocab, tokenizer = build_vocabulary(config.dataset.path, config.dataset.train)

    train_dataloader = instantiate(
        config=config.dataloader,
        dataset_name=config.dataset.train,
        vocab=vocab,
        tokenizer=tokenizer,
    ).create_dataloader()

    val_dataloader = instantiate(
        config=config.dataloader,
        dataset_name=config.dataset.train,
        vocab=vocab,
        tokenizer=tokenizer,
    ).create_dataloader()

    test_dataloader = instantiate(
        config=config.dataloader,
        dataset_name=config.dataset.train,
        vocab=vocab,
        tokenizer=tokenizer,
    ).create_dataloader()

    trainer = instantiate(
        config=config.trainer,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        criterion=criterion,
        optimizer=optimizer
    )

    for epoch in range(config.general.max_epochs):
        trainer.train(current_epoch_nr=epoch)
        trainer.evaluate(current_epoch_nr=epoch, scheduler=scheduler)

    trainer.test()


if __name__ == '__main__':
    main()
