if __name__ == "__main__":
    from datasets import load_dataset
    from pathlib import Path
    import numpy as np 
    from matplotlib import pyplot as plt
    from PIL import Image
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import MinMaxScaler
    from transformers import AdamW
    import tqdm, gc, os, datetime, joblib, glob

    import sys
    sys.path.append("./scripts/")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    date = "20230915"
    dataset = sys.argv[1]
    out_name = f"{date}_resnet_{dataset}"
    out_dir = f"./new/models/{out_name}/"
    logs_dir = f"./new/temp/{out_name}/"

    Path(out_dir + "/scalers").mkdir(parents=True, exist_ok=True)
    Path(logs_dir + "/chkpts").mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    log_file_path = logs_dir + "logs.txt"
    overwrite_logs = False
    if overwrite_logs:
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
    else:
        with open(log_file_path, "a") as f:
            f.write("Starting new run\n at " + str(datetime.datetime.now()) + "\n")

    if os.uname()[1] == "marmalade.physics.upenn.edu":
        print("I'm on marmalade!")
        cache_dir = "/data2/shared/shubh/cache/"
    elif os.uname()[1][:5] == "login" or os.uname()[1][:3] == "nid":
        print("I'm on perlmutter!")
        cache_dir = "/pscratch/sd/s/shubh/ViT/"
        
    data = load_dataset("./data/20230814_224x224/20230814_224x224.py", dataset, cache_dir=cache_dir)
    subset = "train"
    labels = ["H0", "Ob", "Om", "ns", "s8", "w0"]
    size = (224, 224)
    per_device_train_batch_size = 128
    per_device_eval_batch_size = 128
    num_epochs = 200
    learning_rate = 5e-5
    weight_decay_rate = 0.001

    num_channels = np.array(data[subset][0]["map"]).shape[-1]
    
    id2label = {**{labels.index(label):label for label in labels}, \
                **{(len(labels)+labels.index(label)): "ln_sig_" + label for label in labels}}
    label2id = {label:id for id,label in id2label.items()}

    print(id2label)
    print(label2id)

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

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

    def neg_log_likelihood(preds, y):
        # assuming first half of predictions are means and second half are log variances
        means, log_vars = preds[:, :preds.shape[1]//2], preds[:, preds.shape[1]//2:]
        error = y - means
        return torch.mean(0.5 * torch.exp(-log_vars) * error * error + 0.5 * log_vars)

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=len(id2label), bias=True)
    def dropout(model, rate):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                dropout(module, rate)
            if isinstance(module, nn.ReLU):
                new = nn.Sequential(module, nn.Dropout2d(p=rate))
                setattr(model, name, new)
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
    dropout(model, 0.1)

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
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ]
        )

    val_data_augmentation = Compose(
            [
            ]
        )

    def preprocess_train(examples):
        examples["labels"] = np.transpose([examples["scaled_" + x] for x in labels]).astype(np.float32)
        examples['pixel_values'] = [train_data_augmentation(torch.swapaxes(torch.Tensor(np.array(image)), 0, 2)) for image in examples['map']]
        return examples

    def preprocess_val(examples):
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

    train_dataloader = DataLoader(data["train"], collate_fn=collate_fn, batch_size=per_device_train_batch_size)
    val_dataloader = DataLoader(data["validation"], collate_fn=collate_fn, batch_size=per_device_eval_batch_size)
    test_dataloader = DataLoader(data["test"], collate_fn=collate_fn, batch_size=per_device_eval_batch_size)
    
    print("Training model")

    def train_model(model, 
                    data_loader, 
                    val_data_loader,
                    dataset_size, 
                    val_size, 
                    optimizer, 
                    num_epochs):
        model.to(device)
        criterion = neg_log_likelihood

        best_loss = np.inf
        best_epoch = -1
        for epoch in range(num_epochs):
            with open(log_file_path, "a") as f:
                if os.path.exists(logs_dir + f"chkpts/chkpt{epoch}.bin"):
                    # if epoch == 38:
                    model.load_state_dict(torch.load(logs_dir + f"chkpts/chkpt{epoch}.bin"))
                    f.write(f"Loaded checkpoint {epoch}\n")
                    print(f"Loaded checkpoint {epoch}")
                    continue
                else:
                    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                    print('-' * 10)
                    f.write(f"Starting Epoch {epoch}\n")
                    model.train()
                    running_loss = 0.0
                    # Iterate over data.
                    for bi, d in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
                        inputs = d["pixel_values"]
                        labels = d["labels"]
                        inputs = inputs.to(device, dtype=torch.float)
                        labels = labels.to(device, dtype=torch.float)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(True):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        del inputs, labels, outputs, loss, d, bi
                        gc.collect()

                    epoch_loss = running_loss / dataset_size
                    print('Loss: {:.4f}'.format(epoch_loss))
                    f.write(f"Epoch {epoch} Loss: {epoch_loss}\n")
                    del epoch_loss, running_loss
                
                model.eval()
                running_loss = 0.0
                # Iterate over data.
                for bi, d in tqdm.tqdm(enumerate(val_data_loader), total=len(val_data_loader)):
                    inputs = d["pixel_values"]
                    labels = d["labels"]
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.float)

                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    del inputs, labels, outputs, loss, d, bi
                    gc.collect()
                epoch_val_loss = running_loss / val_size
                print('Val Loss: {:.4f}'.format(epoch_val_loss))
                f.write(f"Epoch {epoch} Val Loss: {epoch_val_loss}\n")

                if epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    best_epoch = epoch
                    f.write(f"Best epoch: {best_epoch}\n")
                    print(f"Best epoch: {best_epoch}")
                    torch.save(model.state_dict(), out_dir + "best.bin")
                    np.save(out_dir + "best_epoch.npy", np.array([best_epoch]))
                    
                torch.save(model.state_dict(), logs_dir + f"chkpts/chkpt{epoch}.bin")
                # del epoch_loss, epoch_val_loss, running_loss
                del epoch_val_loss, running_loss
                gc.collect()

        return model
    
    optimizer = AdamW(model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay_rate)
    
    if int(sys.argv[2]) == 0:
        model = train_model(model, train_dataloader, val_dataloader, len(data["train"]), \
                            len(data["validation"]), optimizer, num_epochs)
        torch.save(model.state_dict(), out_dir + "pytorch_model.bin")

    history_files = glob.glob(logs_dir + "/*logs*")

    rn_val_loss = []
    rn_epochs = []
    for history_file in history_files:
        with open(history_file, "r") as f:
            for line in f.readlines():
                if not "Val Loss" in line:
                    continue
                rn_val_loss += [float(line.split()[-1])]
                rn_epochs += [int(line.split()[1])]

    best_epoch = rn_epochs[np.argmin(rn_val_loss)]
    chkpt_path = logs_dir + f"chkpts/chkpt{best_epoch}.bin"
    model.load_state_dict(torch.load(chkpt_path))
    model.to(device)

    with open(out_dir + "preds_model.txt", "w") as f:
        f.write(chkpt_path)

    overwrite_preds = True
    n_pred = 10
    save_after = 10
    for ind in range(n_pred // save_after):
        print(ind)
        if os.path.exists(out_dir + f"preds{ind}.npy") and not overwrite_preds:
            print("skipping iteration", ind)
            continue
        preds = np.empty((save_after, len(data["test"]), len(labels)*2))
        label_ids = []
        for i in range(save_after):
            print(ind, i)
            model.train()
            for bi, d in enumerate(tqdm.tqdm(test_dataloader, total=len(test_dataloader))):
                inputs = d["pixel_values"]
                inputs = inputs.to(device, dtype=torch.float)
                outputs = model(inputs)
                if i == 0:
                    label_ids += d["labels"]
                preds[i, bi*per_device_eval_batch_size:(bi+1)*per_device_eval_batch_size] \
                        = outputs.detach().cpu().numpy()
                del inputs, outputs, d, bi
                gc.collect()
        label_ids = np.array(label_ids)
        np.save(out_dir + f"preds{ind}.npy", preds)
        np.save(out_dir + f"label_ids{ind}.npy", label_ids)
        del preds, label_ids
        gc.collect()

    preds = np.concatenate([np.load(out_dir + f"preds{ind}.npy") for ind in range(n_pred // save_after)], axis=0)
    label_ids = np.concatenate([np.load(out_dir + f"label_ids{ind}.npy") for ind in range(n_pred // save_after)], axis=0)
    print(preds.shape, label_ids.shape)

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
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.2)
    plot_labels = [r"$H_0$", r"$\Omega_b$", r"$\Omega_m$", r"$n_s$", r"$\sigma_8$", r"$w_0$"]
    for ind, (label, ax, low_lim, upp_lim) in enumerate(zip(plot_labels, axs.ravel(), low_lims, upp_lims)):
        p = np.poly1d(np.polyfit(plot_y[:, ind], predictions_best[:, ind], 1))
        ax.errorbar(plot_y[:, ind][::10], predictions_best[:, ind][::10],  predictions_std[:, ind][::10], marker="x", ls='none', alpha=0.4)
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
