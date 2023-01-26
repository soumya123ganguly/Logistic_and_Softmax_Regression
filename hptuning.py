import main

p = [5, 10, 50, 100]
batch_size = [64, 128, 256]
lr = [0.1, 0.01, 0.001]

class HP:
    def __init__(self, batch_size, epochs, learning_rate, k_folds, p):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.k_folds = k_folds
        self.p = p
    def __str__(self):
        return f'Hp(batch_size={self.batch_size}, learning_rate={self.learning_rate}, p={self.p})'

    def __repr__(self):
        return f'Hp(batch_size={self.batch_size}, learning_rate={self.learning_rate}, p={self.p})'

best_hp = None
best_acc = -1
for pi in p:
    for bsj in batch_size:
        for lrk in lr:
            hp = HP(bsj, 100, lrk, 10, pi)
            test_acc = main.main(hp)
            if best_acc < test_acc:
                best_acc = test_acc
                best_hp = hp
                print("Better-HP: {0} {1}".format(best_acc, best_hp))
            else:
                print("not better: {0} {1}".format(test_acc, hp))

print("Best HP: {0} {1}".format(best_acc, best_hp))