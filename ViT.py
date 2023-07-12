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
    from sklearn.preprocessing import MinMaxScaler

    import sys
    sys.path.append("./scripts/")
    from TrainerWithDropout import DropoutTrainer
    from ViTwithNLL import NLLViT

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    date = "20230711"
    # out_name = f"{date}_vit_noisy_2_params"
    out_name = f"{date}_vit_noisy_6_params" + "_no_norm"
    # out_name = f"{date}_vit_6_params"
    out_dir = f"./models/{out_name}/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    subset = "train"
    
    dataset = "noisy"
    num_channels = 40
    # dataset = "noiseless"
    # num_channels = 4
    
    labels = ["H0", "Ob", "Om", "ns", "s8", "w0"]
    # labels = ["Om", "s8"]
    size = (224, 224)
    per_device_train_batch_size = 128
    per_device_eval_batch_size = 256
    num_epochs = 2
    learning_rate = 0.001
    weight_decay_rate = 0.001
    
    id2label = {**{labels.index(label):label for label in labels}, \
                **{(len(labels)+labels.index(label)): "ln_sig_" + label for label in labels}}
    label2id = {label:id for id,label in id2label.items()}

    print(id2label)
    print(label2id)

    data = load_dataset("./data/20230419_224x224/20230419_224x224.py", dataset, cache_dir="/data2/shared/shubh/cache")

    scalers = [MinMaxScaler() for _ in labels]
    for ind, label in enumerate(labels):
        for subset in ["train", "validation", "test"]:
            if subset == "train":
                scalers[ind].fit(np.array(data[subset][label]).reshape(-1, 1))
            scaled_values = scalers[ind].transform(np.array(data[subset][label]).reshape(-1, 1))
            data[subset] = data[subset].add_column("scaled_" + label, scaled_values.reshape(-1)) 

    checkpoint = "google/vit-base-patch16-224-in21k"
    model = NLLViT.from_pretrained(checkpoint,
                problem_type = "regression", id2label=id2label, label2id=label2id, hidden_dropout_prob=0.1,
                num_channels=num_channels, image_size=224, patch_size=16, ignore_mismatched_sizes=True)

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
                # normalize,
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ]
        )

    val_data_augmentation = Compose(
            [
                # normalize,
            ]
        )

    def preprocess_train(examples):
        # examples["labels"] = np.transpose([examples[x] for x in examples.keys() if x != "map"]).astype(np.float32)
        examples["labels"] = np.transpose([examples["scaled_" + x] for x in labels]).astype(np.float32)
        examples['pixel_values'] = [train_data_augmentation(torch.swapaxes(torch.Tensor(np.array(image)), 0, 2)) for image in examples['map']]
        return examples

    def preprocess_val(examples):
        # examples["labels"] = np.transpose([examples[x] for x in examples.keys() if x != "map"]).astype(np.float32)
        examples["labels"] = np.transpose([examples["scaled_" + x] for x in labels]).astype(np.float32)
        examples['pixel_values'] = [val_data_augmentation(torch.swapaxes(torch.Tensor(np.array(image)), 0, 2)) for image in examples['map']]
        return examples

    data["train"].set_transform(preprocess_train)
    data["validation"].set_transform(preprocess_val)
    data["test"].set_transform(preprocess_val)


    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor(np.array([example["labels"] for example in examples]))
        return {"pixel_values": pixel_values, "labels": labels}

    args = TrainingArguments(
        f"./temp/{out_name}",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=num_epochs,
        weight_decay=weight_decay_rate,
        load_best_model_at_end=True,
        logging_dir='logs',
        remove_unused_columns=False,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
    )

    trainer = DropoutTrainer(
        model,
        args,
        train_dataset= data[subset],
        eval_dataset= data["validation"],
        data_collator=collate_fn
    )

    try:
        print("Loading model")
        trainer.model.load_state_dict(torch.load(out_dir + "pytorch_model.bin"))
    except:
        print("Model not found")
    print("Training model")
    trainer.train()
    trainer.save_model(out_dir)

    n_pred = 10
    preds = np.empty((n_pred, len(data["validation"]), len(labels)*2))
    for i in range(n_pred):
        if i % 2 == 0: 
            print(i)
        ps = trainer.predict(data["validation"])
        preds[i] = ps.predictions
        if i == 0:
            label_ids = ps.label_ids

    np.save(out_dir + "preds.npy", preds)
    np.save(out_dir + "label_ids.npy", label_ids)

    preds_best, preds_std = preds[:,:, :preds.shape[-1]//2], preds[:,:, preds.shape[-1]//2:]
    print(preds.shape, preds_best.shape, preds_std.shape)

    predictions = np.empty((preds_best.shape[0] * 100, preds_best.shape[1], preds_best.shape[2]))
    for i in range(preds_best.shape[0]):
        for j in range(100):
            predictions[i*100+j] = np.random.normal(preds_best[i], np.exp(preds_std[i]))

    for ind, scaler in enumerate(scalers):
        predictions[:, :, ind:ind+1] = scaler.inverse_transform(\
            predictions[:, :, ind:ind+1].reshape(-1, 1)).reshape(predictions[:, :, ind:ind+1].shape)
        label_ids[:, ind:ind+1] = scaler.inverse_transform(\
            label_ids[:, ind:ind+1].reshape(-1, 1)).reshape(label_ids[:, ind:ind+1].shape)

    plot_y = label_ids
    predictions_best = np.nanmean(predictions, axis=0)
    predictions_std = np.nanstd(predictions, axis=0)

    upp_lims = np.nanmax(plot_y, axis=0)
    low_lims = np.nanmin(plot_y, axis=0)
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.2)
    # plot_labels = [r"$\Omega_m$", r"$\sigma_8$"]
    plot_labels = [r"$H_0$", r"$\Omega_b$", r"$\Omega_m$", r"$n_s$", r"$\sigma_8$", r"$w_0$"]
    for ind, (label, ax, low_lim, upp_lim) in enumerate(zip(plot_labels, axs.ravel(), low_lims, upp_lims)):
        p = np.poly1d(np.polyfit(plot_y[:, ind], predictions_best[:, ind], 1))
        ax.errorbar(plot_y[:, ind][::10], predictions_best[:, ind][::10],  predictions_std[:, ind][::10], marker="x", ls='none', alpha=0.4)
        # ax.errorbar(plot_y[:, ind][::10], predictions_best[:, ind][::10],  0, marker="x", ls='none', alpha=0.4)
        ax.set_xlabel("true")
        ax.set_ylabel("prediction")
        ax.plot([low_lim, upp_lim], [low_lim, upp_lim], color="black")
        ax.plot([low_lim, upp_lim], [p(low_lim), p(upp_lim)], color="black", ls=":")
        ax.set_xlim([low_lim, upp_lim])
        ax.set_ylim([low_lim, upp_lim])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(label)
        ax.grid()
    plt.savefig(out_dir + "pred-true.png")
    plt.close()