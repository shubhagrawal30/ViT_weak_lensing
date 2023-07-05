if __name__ == "__main__":
    from datasets import load_dataset
    from pathlib import Path
    from transformers import AutoImageProcessor, ViTForImageClassification, ViTConfig, ViTModel
    import numpy as np 
    from matplotlib import pyplot as plt
    from PIL import Image
    import torch
    from transformers import create_optimizer, TrainingArguments, Trainer
    from torch.utils.data import DataLoader
    out_name = "20230620_vit"
    out_dir = f"./models/{out_name}/"
    plot_dir = f"./plots/{out_name}/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    subset = "train"
    labels = ["H0", "Ob", "Om", "ns", "s8", "w0"]
    size = (224, 224)
    batch_size = 128
    num_epochs = 1
    learning_rate = 0.001
    weight_decay_rate = 0.001

    data = load_dataset("./data/20230419_224x224/20230419_224x224.py", cache_dir="/pscratch/sd/s/shubh/")

    id2label = {labels.index(label):label for label in data["train"].features if label in labels}
    label2id = {label:id for id,label in id2label.items()}


    checkpoint = "google/vit-base-patch16-224-in21k"
    model = ViTForImageClassification.from_pretrained(checkpoint,
                problem_type = "regression", id2label=id2label, label2id=label2id, hidden_dropout_prob=0.1,
                num_channels=4, image_size=224, patch_size=16, ignore_mismatched_sizes=True)

    print(model.config.problem_type)
    print(model.config.num_labels)



    from torchvision.transforms import (CenterCrop, 
                                        Compose, 
                                        Normalize, 
                                        RandomHorizontalFlip,
                                        RandomVerticalFlip,
                                        RandomResizedCrop, 
                                        Resize, 
                                        ToTensor)

    normalize = lambda img: (img - torch.mean(img)) / torch.std(img)

    train_data_augmentation = Compose(
            [
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                normalize,
            ]
        )

    val_data_augmentation = Compose(
            [
                normalize,
            ]
        )

    def preprocess_train(examples):
        # examples["labels"] = np.transpose([examples[x] for x in examples.keys() if x != "map"]).astype(np.float32)
        examples["labels"] = np.transpose([examples[x] for x in labels]).astype(np.float32)
        examples['pixel_values'] = [train_data_augmentation(torch.swapaxes(torch.Tensor(np.array(image)), 0, 2)) for image in examples['map']]
        return examples

    def preprocess_val(examples):
        # examples["labels"] = np.transpose([examples[x] for x in examples.keys() if x != "map"]).astype(np.float32)
        examples["labels"] = np.transpose([examples[x] for x in labels]).astype(np.float32)
        examples['pixel_values'] = [val_data_augmentation(torch.swapaxes(torch.Tensor(np.array(image)), 0, 2)) for image in examples['map']]
        return examples

    data["train"].set_transform(preprocess_train)
    data["validation"].set_transform(preprocess_val)
    data["test"].set_transform(preprocess_val)


    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(data[subset], collate_fn=collate_fn, batch_size=batch_size)

    num_train_steps = len(data[subset]) * num_epochs
    optimizer, lr_schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=num_train_steps,
        weight_decay_rate=weight_decay_rate,
        num_warmup_steps=0,
    )

    args = TrainingArguments(
        f"./temp/dropout",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=num_epochs,
        weight_decay=weight_decay_rate,
        load_best_model_at_end=True,
        logging_dir='logs',
        remove_unused_columns=False,
    )

    import sys
    sys.path.append("./scripts/")
    from TrainerWithDropout import DropoutTrainer

    import torch

    trainer = DropoutTrainer(
        model,
        args,
        train_dataset= data[subset],
        eval_dataset= data["validation"],
        data_collator=collate_fn
    )


    trainer.train()


    trainer.save_model(out_dir)

    n_pred = 10
    preds = np.empty((n_pred, len(data["validation"]), 6))
    for i in range(n_pred):
        if i % 2 == 0: 
            print(i)
        ps = trainer.predict(data["validation"])
        preds[i] = ps.predictions
        if i == 0:
            label_ids = ps.label_ids

    plot_y = label_ids
    predictions_best = np.nanmean(preds, axis=0)
    predictions_std = np.nanstd(preds, axis=0)

    upp_lims = np.nanmax(plot_y, axis=0)
    low_lims = np.nanmin(plot_y, axis=0)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.2)
    labels = [r"$\Omega_m$", r"$\sigma_8$"]
    for ind, (label, ax, low_lim, upp_lim) in enumerate(zip(labels, axs.ravel(), low_lims, upp_lims)):
        p = np.poly1d(np.polyfit(plot_y[:, ind], predictions_best[:, ind], 1))
        # ax.errorbar(plot_y[:, ind][::10], predictions_best[:, ind][::10],  predictions_std[:, ind][::10], marker="x", ls='none', alpha=0.4)
        ax.errorbar(plot_y[:, ind][::10], predictions_best[:, ind][::10],  0, marker="x", ls='none', alpha=0.4)
        ax.set_xlabel("true")
        ax.set_ylabel("prediction")
        ax.plot([low_lim, upp_lim], [low_lim, upp_lim], color="black")
        ax.plot([low_lim, upp_lim], [p(low_lim), p(upp_lim)], color="black", ls=":")
        ax.set_xlim([low_lim, upp_lim])
        ax.set_ylim([low_lim, upp_lim])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(label)
        ax.grid()
    plt.savefig(plot_dir + "pred-true.png")
    plt.close()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.2)
    labels = [r"$\Omega_m$", r"$\sigma_8$"]
    for ind, (label, ax) in enumerate(zip(labels, axs.ravel())):
        ax.hist((plot_y[:, ind] - predictions_best[:, ind]) / predictions_best[:, ind], bins=100, density=True, histtype="step")
        ax.set_title(label)
        ax.grid()
        ax.set_xlim([-0.7, 0.7])
    plt.savefig(plot_dir + "hist.png")
    plt.close()