from antiproton_flux import dphidlogK
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from timeit import default_timer as timer

plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

style = ['--', '-', '-.']
color = ['tab:blue', 'tab:orange', 'tab:green',
         'tab:red', 'tab:purple', 'tab:brown']


def ffnn_model():
    model = Sequential()
    model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='tanh', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='tanh', kernel_initializer='he_uniform'))
    return model


def generate_input(K_min, K_max, N_K, M, channel, cross_section, profile, V_conv, L, K_diff_0, delta):
    K = np.logspace(K_min, K_max, N_K)
    y = np.array([dphidlogK(K_, M, channel, cross_section, profile, V_conv,
                            L, K_diff_0, delta) for K_ in K])
    K = np.log10(K)
    y = np.log10(y)

    K = K.reshape(-1, 1)
    y = y.reshape(-1, 1)
    parameter = M*np.ones(N_K).reshape(-1, 1)

    X = np.hstack((K, parameter))

    return X, y


channel = 'b'
cross_section = 1e-26
profile = 'Iso'
L = [1, 4, 15]                          # kpc
V_conv = [13.5, 12, 5]                  # km/s
K_diff_0 = [0.0016, 0.0112, 0.0765]     # kpc^2/Myr
delta = [0.85, 0.70, 0.46]
profiles = ['Iso', 'Bur', 'NFW', 'Moo', 'Ein', 'EiB']
i = 1


K_min = -1
K_max = 4
N_K = 50
M_min = 10
M_max = 1000
N_M = 50

X = []
y = []

start = timer()
for M in np.linspace(M_min, M_max, N_M):
    X_, y_ = generate_input(K_min, K_max, N_K, M, channel, cross_section,
                            profile, V_conv[i], L[i], K_diff_0[i], delta[i])
    X.append(X_)
    y.append(y_)
end = timer()
generate_time = end - start

X = np.array(X).reshape(-1, 2)
y = np.array(y).reshape(-1, 1)

scale_X = MinMaxScaler()
X_scaled = scale_X.fit_transform(X)

scale_y = MinMaxScaler()
y_scaled = scale_y.fit_transform(y)

model = ffnn_model()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

model.compile(loss='mse', optimizer='adam')

start = timer()
history = model.fit(X_train, y_train, epochs=1000, batch_size=100, verbose=1)
end = timer()
training_time = end - start
test_loss = model.evaluate(X_test, y_test, verbose=0)
last_loss = history.history['loss'][-1]

print('generate time: {} s'.format(generate_time))
print('training time: {} s'.format(training_time))
print('last loss: {}'.format(last_loss))
print('test loss: {}'.format(test_loss))

mse_pred = []
gen_time = []
pred_time = []
for c in range(20):
    M = np.random.uniform(M_min, M_max)
    start = timer()
    X, y = generate_input(K_min, K_max, N_K, M, channel, cross_section,
                          profile, V_conv[i], L[i], K_diff_0[i], delta[i])
    end = timer()
    gen_time.append(end-start)
    X = scale_X.transform(X)
    y = scale_y.transform(y)
    start = timer()
    yhat = model.predict(X)
    end = timer()
    pred_time.append(end-start)
    mse_pred.append(mean_squared_error(y, yhat))
mse_pred_mean = np.mean(mse_pred)
gen_time_mean = np.mean(gen_time)
pred_time_mean = np.mean(pred_time)
print('prediction MSE: {}'.format(mse_pred_mean))
print('gen time: {} s'.format(gen_time_mean))
print('pred time: {} s'.format(pred_time_mean))


fig, ax = plt.subplots(figsize=(6.4, 4.8))

M = [50, 100, 200, 500, 1000]
for c, M in enumerate(M):
    X, y = generate_input(K_min, K_max, N_K, M, channel, cross_section,
                          profile, V_conv[i], L[i], K_diff_0[i], delta[i])

    X = scale_X.transform(X)
    y = scale_y.transform(y)

    yhat = model.predict(X)

    X = scale_X.inverse_transform(X)
    y = scale_y.inverse_transform(y)
    yhat = scale_y.inverse_transform(yhat)

    ax.plot(np.power(10, X[:, 0]), np.power(10, y), color=color[c], label=(
        r'$M_{\mathrm{DM}}$' + ' = {} GeV'.format(M)))
    ax.plot(np.power(10, X[:, 0]), np.power(10, yhat),
            marker='.', linestyle='none', color=color[c])

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([1e-6, 3e-2])
ax.set_xlim([9e-2, 1e3])
ax.set_ylabel(r'$d\Phi / (d \log(K/GeV))$ [1/m$^2$ s sr]')
ax.set_xlabel('$K$ [GeV]')

ax.legend()

fig.tight_layout()
plt.savefig('figures/ffnn_mas.pdf')
