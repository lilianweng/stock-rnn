import numpy as np
import os
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_path(rel_path):
    return os.path.join(REPO_ROOT, rel_path)


def recover_normalized_prices(prices: np.ndarray, init_denom: float = 1.):
    """
    [1, 2] => [0, 1]
    [3, 4] => [3/2 - 1, 4/2 - 1]
    [5, 6] => [5/4 - 1, 6/4 - 1]
    """
    assert len(prices.shape) == 2
    prices /= 100.  # back to ratio.
    recovered = []
    denom = init_denom

    for inp in prices:
        # print(inp, denom, "==>", (inp + 1.) * denom)
        recovered.append((inp + 1.) * denom)
        denom = recovered[-1][-1]

    recovered = np.array(recovered)
    assert prices.shape == recovered.shape
    return recovered


def plot_prices(norm_preds: np.ndarray, norm_truths: np.ndarray, image_path: str,
                stock_sym: str = None):
    assert norm_preds.shape == norm_truths.shape

    truths = recover_normalized_prices(norm_truths)
    preds = recover_normalized_prices(norm_preds)

    norm_truths = norm_truths.flatten()
    norm_preds = norm_preds.flatten()
    truths = truths.flatten()
    preds = preds.flatten()
    days = range(len(truths))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), tight_layout=True)

    ax1.plot(days, norm_truths, label='truth', color='k', alpha=0.5)
    ax1.plot(days, norm_preds, label='prediction', color='r')
    ax1.set_xlabel('day')
    ax1.set_ylabel('normalized price')
    ax1.grid(color='k', ls=':', alpha=0.3)
    ax1.legend(loc='best', frameon=False)

    ax2.plot(days, truths, label='truth', color='k', alpha=0.5)
    ax2.plot(days, preds, label='prediction', color='r')
    ax2.set_xlabel('day')
    ax2.set_ylabel('price')
    ax2.grid(color='k', ls=':', alpha=0.3)
    ax2.legend(loc='best', frameon=False)

    if stock_sym:
        plt.title(stock_sym + " | Last %d days in test" % len(truths))

    plt.savefig(image_path, format='png', bbox_inches='tight', transparent=True)
    plt.close()
