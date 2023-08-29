import numpy as np
import torch, sys, time
from sbi.inference import SNPE
from sbi import utils as utils
import multiprocessing as mp
from itertools import repeat

def get_one_chain(args):
    start = time.time()
    n, idx, preds, labels, num_samples, low, high = args
    print(f"{n}: chain {idx}")
    x0, true = preds[idx], labels[idx]

    inds = np.delete(np.arange(preds.shape[0]), idx)
    preds, labels = preds[inds], labels[inds]

    prior = utils.BoxUniform(low=low, high=high)
    inference = SNPE(prior=prior, show_progress_bars=True, \
                     density_estimator=utils.posterior_nn(model='maf',
                                hidden_features=50, num_transforms=4))
    theta, x = torch.FloatTensor(labels), torch.FloatTensor(preds)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)

    posterior_samples = posterior.set_default_x(x0).sample((num_samples,), \
                                                x=x0, show_progress_bars=False)
    print(f"chain {idx} done!", time.time()-start)

    return posterior_samples.numpy()

if __name__ == "__main__":
    model_name = sys.argv[1]
    preds_labels_dir = f"../models/{model_name}/"
    out_file = f"../chains/{model_name}.npz"
    num_preds = 1
    para_idx = (2, 4, 5)
    # num_chains = 200
    n_threads = 16
    num_samples = 10000

    try:
        preds = np.concatenate([np.load(preds_labels_dir + f"preds{i}.npy") \
                                for i in range(num_preds)], axis=0)
        labels = np.concatenate([np.load(preds_labels_dir + f"label_ids{i}.npy") \
                                for i in range(num_preds)], axis=0)
    except:
        preds = np.load(preds_labels_dir + "preds.npy")
        labels = np.load(preds_labels_dir + "label_ids.npy")

    preds = np.mean(preds[:, :, :preds.shape[-1]//2], axis=0)
    preds, labels = preds[:, para_idx], labels[:, para_idx]

    # chain_idx = np.random.choice(np.arange(preds.shape[0]), size=num_chains)
    chain_idx = np.arange(preds.shape[0])
    num_chains = len(chain_idx)

    low, high = np.min(labels, axis=0), np.max(labels, axis=0)

    
    args = zip(np.arange(num_chains), chain_idx, repeat(preds), repeat(labels), \
               repeat(num_samples), repeat(low), repeat(high))
    if False:
        pool = mp.pool.ThreadPool(n_threads)
        results = pool.map(get_one_chain, args)
        pool.close()
        pool.join()
    else:
        results = np.array(list(map(get_one_chain, args)))

    np.savez(out_file, chains=results, preds=preds[chain_idx], labels=labels[chain_idx])

    print("done!")