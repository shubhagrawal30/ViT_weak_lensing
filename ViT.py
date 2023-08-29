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

    import sys, os, gc, tqdm, joblib, json
    sys.path.append("./scripts/")
    from TrainerWithDropout import DropoutTrainer
    from ViTwithNLL import NLLViT

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    date = "20230711"
    # date = "20230810"
    out_name = f"{date}_vit_noisy_6_params" + "_no_norm"
    # dataset = ["DES_one_bin", "DES_half_sky", "DES"][int(sys.argv[1])]
    # out_name = f"{date}_vit_{dataset}"
    # out_name = f"{date}_vit_DES_half_sky"
    # out_name = f"{date}_vit_DES"
    out_dir = f"./models/{out_name}/"
    Path(out_dir + "scalers/").mkdir(parents=True, exist_ok=True)

    normalize = lambda img: img
    # normalize = lambda img: (img - torch.mean(img)) / torch.std(img)

    subset = "train"
    
    # num_channels = {"DES": 40, "DES_half_sky": 20, "DES_one_bin": 10, \
    #                 "noisy": 40, "noiseless": 4}[dataset]
    dataset = "noisy"
    # dataset = "noiseless"
    num_channels = 40
    
    labels = ["H0", "Ob", "Om", "ns", "s8", "w0"]
    # labels = ["Om", "s8"]
    size = (224, 224)
    per_device_train_batch_size = 128
    per_device_eval_batch_size = 256
    num_epochs = 300
    learning_rate = 5e-5
    weight_decay_rate = 0.001
    
    id2label = {**{labels.index(label):label for label in labels}, \
                **{(len(labels)+labels.index(label)): "ln_sig_" + label for label in labels}}
    label2id = {label:id for id,label in id2label.items()}

    print(id2label)
    print(label2id)
    print(dataset, num_channels)

    data = load_dataset("./data/20230419_224x224/20230419_224x224.py", dataset, cache_dir="/data2/shared/shubh/cache")

    try:
        print("trying to load scalers")
        scalers = [joblib.load(out_dir + "scalers/" + label + ".pkl") for label in labels]
        for ind, label in enumerate(labels):
            for subset in ["train", "validation", "test"]:
                scaled_values = scalers[ind].transform(np.array(data[subset][label]).reshape(-1, 1))
                data[subset] = data[subset].add_column("scaled_" + label, scaled_values.reshape(-1))
    except:
        print("scalers not found")
        scalers = [MinMaxScaler() for _ in labels]
        for ind, label in enumerate(labels):
            for subset in ["train", "validation", "test"]:
                if subset == "train":
                    scalers[ind].fit(np.array(data[subset][label]).reshape(-1, 1))
                scaled_values = scalers[ind].transform(np.array(data[subset][label]).reshape(-1, 1))
                data[subset] = data[subset].add_column("scaled_" + label, scaled_values.reshape(-1)) 
        print("saving scalers")
        for ind, scaler in enumerate(scalers):
            joblib.dump(scaler, out_dir + "scalers/" + labels[ind] + ".pkl")


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

    train_data_augmentation = Compose(
            [
                normalize,
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ]
        )

    val_data_augmentation = Compose(
            [
                normalize,
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
        trainer.train(resume_from_checkpoint=True)
    except:
        trainer.train()
    trainer.save_model(out_dir)

    # try:
    #     print("Loading model")
    #     # trainer.model.load_state_dict(torch.load(out_dir + "pytorch_model.bin"))
    chkpts = os.listdir("./temp/" + out_name)
    chkpts = [int(chkpt.split("-")[1]) for chkpt in chkpts if "checkpoint" in chkpt]
    highest_chkpt = max(chkpts)
    with open(f"./temp/{out_name}/checkpoint-{highest_chkpt}/trainer_state.json", "r") as f:
        checkpoint_path = json.load(f)['best_model_checkpoint'] + "/pytorch_model.bin"

    # print("Loading model from", checkpoint_path)
    # trainer.model.load_state_dict(torch.load(checkpoint_path))
    # except:
    # print("Model not found")
    # print("Training model")

    with open(out_dir + "preds_model.txt", "w") as f:
        f.write(checkpoint_path)

    n_pred = 10
    save_after = 10
    for ind in range(n_pred // save_after):
        print(ind)
        if os.path.exists(out_dir + f"preds{ind}.npy"):
            print("skipping iteration", ind)
            continue
        preds = np.empty((save_after, len(data["test"]), len(labels)*2))
        for i in range(save_after):
            print(ind, i)
            ps = trainer.predict(data["test"])
            preds[i] = ps.predictions
            if i == 0:
                label_ids = ps.label_ids
        np.save(out_dir + f"preds{ind}.npy", preds)
        np.save(out_dir + f"label_ids{ind}.npy", label_ids)

    preds = np.concatenate([np.load(out_dir + f"preds{i}.npy") for i in range(n_pred // save_after)], axis=0)
    label_ids = np.concatenate([np.load(out_dir + f"label_ids{i}.npy") for i in range(n_pred // save_after)], axis=0)
    print(preds.shape, label_ids.shape)

    preds_best, preds_std = preds[:,:, :preds.shape[-1]//2], preds[:,:, preds.shape[-1]//2:]
    print(preds.shape, preds_best.shape, preds_std.shape)

    n_preds_per_sample = 100
    predictions = np.empty((preds_best.shape[0] * n_preds_per_sample, preds_best.shape[1], preds_best.shape[2]))
    for i in range(preds_best.shape[0]):
        for j in range(n_preds_per_sample):
            predictions[i*n_preds_per_sample+j] = np.random.normal(preds_best[i], np.exp(preds_std[i]))

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

    with open(out_dir + "preds_model.txt", "a") as f:
        f.write("\n" + out_dir + "pred-true.png")