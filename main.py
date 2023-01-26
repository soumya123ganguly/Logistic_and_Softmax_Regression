import argparse
import network
import data
import image
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

def main(hyperparameters):
    # Initializing PCA, train and test sets.
    pca = PCA(n_components = hyperparameters.p)
    train_data, train_labels = data.load_data("data/")
    # Commented for computing the train indices for classifying digits 2, 7 and 5, 8
    #idx27 = np.where(np.logical_or(train_labels == 2, train_labels == 7))
    #train_data = train_data[idx27]
    #train_labels = train_labels[idx27]
    #train_labels = np.where(train_labels == 2, 1, 0).reshape(-1, 1)
    train_labels = data.onehot_encode(train_labels)
    train_data = train_data.reshape(len(train_data), -1)
    test_data, test_labels = data.load_data("data/", train=False)
    # Commented for computing the test indices for classifying digits 2, 7 and 5, 8
    #idx27 = np.where(np.logical_or(test_labels == 2, test_labels == 7))
    #test_data = test_data[idx27]
    #test_labels = test_labels[idx27]
    #test_labels = np.where(test_labels == 2, 1, 0).reshape(-1, 1)
    test_labels = data.onehot_encode(test_labels)
    test_data = test_data.reshape(len(test_data), -1)
    # transformaing the train and test set using the PCA components
    train_data_pca = pca.fit_transform(train_data)
    test_data_pca = pca.transform(test_data)

    # Initializing best validation accuracy and train losses and val losses to be used in plots
    best_val_acc = -1
    best_mean, best_std = 0, 0
    train_losses = np.zeros((hyperparameters.epochs,))
    val_losses = np.zeros((hyperparameters.epochs,))

    # Initialize the k-fold dataset iterator.
    k_fold_iter = data.generate_k_fold_set((train_data_pca, train_labels), k=hyperparameters.k_folds)
    for _ in range(hyperparameters.k_folds):
        # Dataset preprocessing.
        train_data_pca, train_labels = data.shuffle((train_data_pca, train_labels))
        train_folds, val_fold = next(k_fold_iter)
        _, mean, std = data.z_score_normalize(train_folds[0])
        train_folds = (data.append_bias(data.z_score_normalize(train_folds[0])[0]), train_folds[1])
        val_fold = (data.append_bias(data.z_score_normalize(val_fold[0])[0]), val_fold[1])
        test_data_temp = data.append_bias(data.z_score_normalize(test_data_pca)[0])
        # Network initialization.
        net = network.Network(hyperparameters, network.softmax, 
                              network.multiclass_cross_entropy, 10)
        for epoch in tqdm(range(hyperparameters.epochs)):
                train_acc, train_loss = 0, 0
                # train and validation folds minibatch iterator initialization.
                mb_train_iter = data.generate_minibatches(train_folds, batch_size=hyperparameters.batch_size)
                mb_val_iter = data.generate_minibatches(val_fold, batch_size=hyperparameters.batch_size)
                for i in range(len(train_folds[0])//hyperparameters.batch_size):
                        mb_train_folds = next(mb_train_iter)
                        train_loss_batch, train_acc_batch = net.train(mb_train_folds)
                        train_acc += train_acc_batch
                        train_loss += train_loss_batch
                # Compute train loss and accuracy
                train_acc /= (len(train_folds[0])//hyperparameters.batch_size)
                train_loss /= (len(train_folds[0])//hyperparameters.batch_size)
                val_acc, val_loss = 0, 0
                for _ in range(len(val_fold[0])//hyperparameters.batch_size):
                        mb_val_fold = next(mb_val_iter)
                        val_loss_batch, val_acc_batch = net.test(mb_val_fold)
                        val_acc += val_acc_batch
                        val_loss += val_loss_batch
                # Compute train loss and accuracy
                val_acc /= (len(val_fold[0])//hyperparameters.batch_size)
                val_loss /= (len(val_fold[0])//hyperparameters.batch_size)
                best_val_acc = max(best_val_acc, val_acc)
                # Choosing the mean, standard deviation and best validation accuracy
                if best_val_acc == val_acc:
                        best_mean, best_std = mean, std
                train_losses[epoch] += train_loss
                val_losses[epoch] += val_loss
    # Compute test accuracy
    test_acc, test_loss = 0, 0
    test_iter = data.generate_minibatches((test_data_temp, test_labels), batch_size=hyperparameters.batch_size)
    for _ in range(len(test_data_temp)//hyperparameters.batch_size):
            mb_test_folds = next(test_iter)
            test_loss_batch, test_acc_batch = net.train(mb_test_folds)
            test_acc += test_acc_batch
            test_loss += test_loss_batch
    test_acc /= (len(test_data_temp)//hyperparameters.batch_size)
    test_loss /= (len(test_data_temp)//hyperparameters.batch_size)
    train_losses /= hyperparameters.k_folds
    val_losses /= hyperparameters.k_folds
    print("test accuracy: {0} validation accuracy: {1}".format(test_acc, best_val_acc))
    # Compute the train and validation loss curves.
    plot.xlabel('Epochs')
    plot.ylabel('Loss')
    plot.title('Average Loss for 10 digit classification with \n PCA features:{0} Learning Rate:{1} Batch Size:{2} k:{3}'.format(hyperparameters.p, hyperparameters.learning_rate, hyperparameters.batch_size, hyperparameters.k_folds))
    plot.plot(np.arange(len(train_losses)), train_losses, c='r', label='Train')
    plot.plot(np.arange(len(val_losses)), val_losses, c='b', label='Validation')
    plot.legend()
    plot.savefig('101.png')
    return best_val_acc 


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
