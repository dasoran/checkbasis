import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import math
import matplotlib.pyplot as plt

# define constance
n_sampling_rate = 100


# create training and evaluation datas
# n_train_batchset
# n_test_batchset
# x_train
# x_test
def create_data_with_fourier_basis():
    datas = []
    params = []
    for x in range(0, 2 * 10):
        params.append(np.random.rand())
    for raw_x in range(0, n_sampling_rate):
        x = raw_x / n_sampling_rate
        data = 0
        for i in range(0, len(params), 2):
            data += params[i]     * math.sin(2 * math.pi * (i + 1) * x)
            data += params[i + 1] * math.cos(2 * math.pi * (i + 1) * x)
        datas.append(data)
    return datas

n_train_batchset = 10000
n_test_batchset = 10000
raw_x_train = []
raw_x_test = []
for i in range(0, n_train_batchset):
    raw_x_train.extend(create_data_with_fourier_basis())
for i in range(0, n_test_batchset):
    raw_x_test.extend(create_data_with_fourier_basis())
np_x_train = np.array(raw_x_train, dtype=np.float32)
x_train = np_x_train.reshape((n_train_batchset, n_sampling_rate))
np_x_test = np.array(raw_x_test, dtype=np.float32)
x_test = np_x_test.reshape((n_test_batchset, n_sampling_rate))

print('generated datas')


# define model constance
n_input = n_sampling_rate
n_units = 100

n_epoch = 100

batchsize = 100


# define model
model = FunctionSet(
    l1 = F.Linear(n_input, n_units),
    l2 = F.Linear(n_units, n_input)
)

#def forward(x_data, train = True):
#    x, t = Variable(x_data), Variable(x_data)
#    h1 = F.dropout(F.relu(model.l1(x)), train = train)
#    y = F.dropout(model.l2(h1), train = train)
#    return F.mean_squared_error(t, y)
def forward(x_data, train = True):
    x, t = Variable(x_data), Variable(x_data)
    h1 = F.dropout(model.l1(x), train = train)
    y = F.dropout(model.l2(h1), train = train)
    return F.mean_squared_error(t, y)


optimizer = optimizers.Adam()
optimizer.setup(model)


log_loss_train = []
log_loss_test = []
# training loop
for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(n_train_batchset)
    sum_loss = 0
    for i in range(0, n_train_batchset, batchsize):
        x_batch = np.asarray(x_train[perm[i:i + batchsize]])

        optimizer.zero_grads()
        loss = forward(x_batch, train=False)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(x_batch)

    print('train mean loss={}'.format(sum_loss / n_train_batchset))
    log_loss_train.append(sum_loss / n_train_batchset)

    # evaluation
    sum_loss = 0
    for i in range(0, n_test_batchset, batchsize):
        x_batch = np.asarray(x_test[i:i + batchsize])

        loss = forward(x_batch, train=False)

        sum_loss += float(loss.data) * len(x_batch)

    print('test  mean loss={}'.format(sum_loss / n_test_batchset))
    log_loss_test.append(sum_loss / n_test_batchset)

# do that you want to after training

x_range = np.arange(0, n_epoch, 1)
plt.plot(x_range, log_loss_train, label='train')
plt.plot(x_range, log_loss_test, label='test')
plt.legend()
plt.show()

plt.clf()
raw_data = np.array(create_data_with_fourier_basis(), dtype=np.float32)
data = raw_data.reshape((1, n_sampling_rate))

x, t = Variable(data), Variable(data)
h1 = F.dropout(F.relu(model.l1(x)), train = False)
y = F.dropout(model.l2(h1), train = False)

print(h1.data)

x_range = np.arange(0, n_sampling_rate, 1)
plt.plot(x_range, data[0], label='source')
plt.plot(x_range, y.data[0], label='result')
plt.legend()
plt.show()

for i in range(0, n_units):
    plt.clf()
    plt.plot(x_range, model.l1.W[i])
    plt.savefig('img/{0:03d}.png'.format(i))

l2_reverse = []
for i in range(0, n_input):
    for j in range(0, n_units):
        if i == 0:
            l2_reverse.append([])
        l2_reverse[j].append(model.l2.W[i][j])

print(len(l2_reverse[0]))

#for i in range(0, n_units):
#    plt.clf()
#    plt.plot(x_range, l2_reverse[i])
#    plt.savefig('img/l2_{0:03d}.png'.format(i))



