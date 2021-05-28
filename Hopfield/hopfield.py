from hopfieldnetwork import HopfieldNetwork
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_data():
    five = './bitmap/5_perfect.csv'
    df = pd.read_csv(five, header=None)
    df.head()
    five = df.values

    six = './bitmap/6_perfect.csv'
    df = pd.read_csv(six, header=None)
    df.head()
    six = df.values

    eight = './bitmap/8_perfect.csv'
    df = pd.read_csv(eight, header=None)
    df.head()
    eight = df.values

    nine = './bitmap/9_perfect.csv'
    df = pd.read_csv(nine, header=None)
    df.head()
    nine = df.values

    return [five, six, eight, nine]


def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted


def preprocessing(img, w=11, h=11):
    # Reshape
    flatten = np.reshape(img, (w*h))
    return flatten


def reshape(data):
    data = np.reshape(data, (11, 11))
    return data


def plot(data, test, predicted, figsize=(5, 6)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i == 0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')

        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()


def plot_single(data, test, pred):
    data = reshape(data)
    test = reshape(test)
    pred = reshape(pred)
    fig, axarr = plt.subplots(1, 3)
    axarr[0].imshow(data)
    axarr[0].axis('off')
    axarr[1].imshow(test)
    axarr[1].axis('off')
    axarr[2].imshow(pred)
    axarr[2].axis('off')
    axarr[0].set_title('Train data')
    axarr[1].set_title("Input data")
    axarr[2].set_title('Output data')

    plt.tight_layout()
    plt.show()


def main():
    hn = HopfieldNetwork(N=121)

    data = load_data()
    test = []
    out = []

    # save patterns in network
    for input_pattern in data:
        hn.train_pattern(preprocessing(input_pattern))
        hn.compute_energy(preprocessing(input_pattern))
        hn.save_network("network")

    # test network
    mode = 'async' if int(
        input("Mode:\n1. Async\n2. Sync\n") or 1) == 1 else 'sync'
    for input_pattern in data:
        hn = HopfieldNetwork(filepath="network.npz")
        test_pattern = get_corrupted_input(preprocessing(input_pattern), 0.05)
        test.append(test_pattern)
        hn.train_pattern(test_pattern)
        while(not hn.check_stability(hn.S)):
            hn.update_neurons(iterations=100, mode=mode)
            print("Βρίσκεται σε κατάσταση ισοροπίας? %s" %
                  str(hn.check_stability(hn.S)))
        plot_single(input_pattern, test_pattern, hn.S)
        out.append(hn.S)

    plot(data, test, out)


if __name__ == '__main__':
    main()
