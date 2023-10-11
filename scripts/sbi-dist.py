print("starting")
import numpy as np
from matplotlib import pyplot as plt
import torch
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import getdist
from getdist import plots, MCSamples
import sys, pathlib
print("done importing")

model_names = sys.argv[2:]
date = "20230915"
folder = f"../new/models/{date}_"
out_dir = f"../new/plots/{sys.argv[1]}/"
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

indices = [2, 4, 5]
index = 1590
pars = np.array(["H0", "Ob", "Om", "ns", "s8", "w0"])
colors = ["blue", "red", "purple", "orange", "green", "brown", \
    "pink", "gray", "olive", "cyan"][:len(model_names)]

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
gd_samples = []

for ind, model in enumerate(model_names):
    print("Model: ", model)
    preds = np.load(f"{folder}{model}/preds0.npy")
    labels = np.load(f"{folder}{model}/label_ids0.npy")
    preds = np.mean(preds[:, :, :preds.shape[2]//2], axis=0)
    
    for i in range(len(pars)):
        ax = axs[i//3, i%3]
        ax.scatter(labels[:, i], preds[:, i], color=colors[ind], \
            label=model, marker="x", s=1, alpha=0.5)
        ran = np.min(labels[:, i]), np.max(labels[:, i])
        ax.plot(ran, ran, c="black", ls="--", lw=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(pars[i])
        ax.grid(True)
        ax.set_aspect("equal")
        
    if index is None:
        index = np.random.randint(0, len(labels))
        
    preds, labels = preds[:, indices], labels[:, indices]
    x_0 = preds[index]
    true = labels[index]
    indexs = np.delete(np.arange(len(labels)), index)
    labels, preds = labels[indexs], preds[indexs]
    
    prior = utils.BoxUniform(low=np.min(labels, axis=0), \
            high=np.max(labels, axis=0))
    inference = SNPE(prior=prior, density_estimator=utils.posterior_nn(\
                            model='maf', hidden_features=50, num_transforms=4))
    theta, x = torch.FloatTensor(labels), torch.FloatTensor(preds)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    
    samples = posterior.set_default_x(x_0).sample((10000,), x=x_0)
    
    gd_samples.append(MCSamples(samples=samples.numpy(), names=pars[indices], \
        labels=pars[indices], ranges={pars[i]: (np.min(labels[:, i]), \
        np.max(labels[:, i])) for i in range(len(indices))}, label=model))
    
plt.figure(fig.number)
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_dir}predtrue.png")
plt.close()

g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, filled=False, contour_colors=colors)

# add truth values to get subplot
for i in range(len(indices)):
    for j in range(len(indices)):
        if j > i:
            continue
        ax = g.subplots[i, j]
        if i == j:
            ax.axvline(true[i], color="black", ls="--", lw=1)
        else:
            ax.scatter(true[j], true[i], color="black", marker="x", s=20) 
        ax.grid(True, ls="--", lw=1, alpha=0.5)

g.export(f"{out_dir}triangle.png")

print("done!")