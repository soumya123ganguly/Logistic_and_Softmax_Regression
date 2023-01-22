import argparse
import network
import data
import image
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

def main(hyperparameters):
    pca = PCA(n_components = hyperparameters.p)
    train_data, train_labels = data.load_data("data/")
    #idx27 = np.where(np.logical_or(train_labels == 2, train_labels == 7))
    #train_data = train_data[idx27]
    #train_labels = train_labels[idx27]
    #train_labels = np.where(train_labels == 2, 1, 0).reshape(-1, 1)
    train_labels = data.onehot_encode(train_labels)
    train_data = train_data.reshape(len(train_data), -1)
    test_data, test_labels = data.load_data("data/", train=False)
    #idx27 = np.where(np.logical_or(test_labels == 2, test_labels == 7))
    #test_data = test_data[idx27]
    #test_labels = test_labels[idx27]
    #test_labels = np.where(test_labels == 2, 1, 0).reshape(-1, 1)
    test_labels = data.onehot_encode(test_labels)
    test_data = test_data.reshape(len(test_data), -1)
    train_data_pca = pca.fit_transform(train_data)
    test_data_pca = pca.transform(test_data)

    k_fold_iter = data.generate_k_fold_set((train_data_pca, train_labels), k=hyperparameters.k_folds)
    for fold in range(hyperparameters.k_folds):
        train_data_pca, train_labels = data.shuffle((train_data_pca, train_labels))
        train_folds, val_fold = next(k_fold_iter)
        net = network.Network(hyperparameters, network.softmax, 
                              network.multiclass_cross_entropy, 10)
        best_val_acc = -1
        train_losses = []
        val_losses = []
        for _ in tqdm(range(hyperparameters.epochs)):
                train_acc, train_loss = 0, 0
                # mb_train_prev = None
                mb_train_iter = data.generate_minibatches(train_folds, batch_size=hyperparameters.batch_size)
                mb_val_iter = data.generate_minibatches(val_fold, batch_size=hyperparameters.batch_size)
                for i in range(len(train_folds[0])//hyperparameters.batch_size):
                        mb_train_folds = next(mb_train_iter)
                        train_loss_batch, train_acc_batch = net.train(mb_train_folds)
                        train_acc += train_acc_batch
                        train_loss += train_loss_batch
                train_acc /= (len(train_folds[0])//hyperparameters.batch_size)
                train_loss /= (len(train_folds[0])//hyperparameters.batch_size)
                val_acc, val_loss = 0, 0
                for _ in range(len(val_fold[0])//hyperparameters.batch_size):
                        mb_val_fold = next(mb_val_iter)
                        val_loss_batch, val_acc_batch = net.test(mb_val_fold)
                        val_acc += val_acc_batch
                        val_loss += val_loss_batch
                val_acc /= (len(val_fold[0])//hyperparameters.batch_size)
                val_loss /= (len(val_fold[0])//hyperparameters.batch_size)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
        test_acc, test_loss = 0, 0
        test_iter = data.generate_minibatches((test_data_pca, test_labels))
        for _ in range(len(test_data_pca)//hyperparameters.batch_size):
                mb_test_folds = next(test_iter)
                test_loss_batch, test_acc_batch = net.train(mb_test_folds)
                test_acc += test_acc_batch
                test_loss += test_loss_batch
        test_acc /= (len(test_data_pca)//hyperparameters.batch_size)
        test_loss /= (len(test_data_pca)//hyperparameters.batch_size)
        plot.plot(np.arange(len(train_losses)), train_losses)
        plot.plot(np.arange(len(val_losses)), val_losses)
        plot.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'CSE151B PA1')
    parser.add_argument('--batch-size', type = int, default = 1,
            help = 'input batch size for training (default: 1)')
    parser.add_argument('--epochs', type = int, default = 100,
            help = 'number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type = float, default = 0.001,
            help = 'learning rate (default: 0.001)')
    parser.add_argument('--z-score', dest = 'normalization', action='store_const', 
            default = data.min_max_normalize, const = data.z_score_normalize,
            help = 'use z-score normalization on the dataset, default is min-max normalization')
    parser.add_argument('--k-folds', type = int, default = 5,
            help = 'number of folds for cross-validation')
    parser.add_argument('--p', type = int, default = 100,
            help = 'number of principal components')

    hyperparameters = parser.parse_args()
    main(hyperparameters)
