import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GRU, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import eli5
from eli5.sklearn import PermutationImportance

# cp.vl["EPS_STATUS"]['LKA_STATE'] status meanings
# 0: stopped? in park? it's 0 when vehicle first starts

# 1: steer req is 0, torque is not being commanded or applied. generally ok and fault-free
# 5: engaged and applying torque, no faults

# 9: steer fault ocurred, not applying torque for duration. 200 - 4 frames of 9 after fault

# 21: unknown, but status has been 21 rarely
# 25: on rising edge of a steering fault, occurs for 4 frames

ok_states = [0, 1, 5]
after_fault = 9

with open('data', 'rb') as f:
  fault_data = pickle.load(f)


faults = []
non_faults = []
non_fault_look_ahead = 100  # in frames
fault_look_ahead = 4  # in frames

for data in fault_data:
  for i, line in enumerate(data):
    line['angle_steers'] = line['can']['angle_steers']
    line['torque_cmd'] = line['can']['torque_cmd']
    line['driver_torque'] = line['can']['driver_torque']
    if i + non_fault_look_ahead >= len(data):
      continue
    # line['angle_accel'] = (line['can']['angle_steers'] - data[i - 1]['can']['angle_steers'])

    if line['can']['lka_state'] == 5 and data[i + fault_look_ahead]['can']['lka_state'] == 25 and data[i + fault_look_ahead - 1]['can']['lka_state'] == 5:  # catch faults only at frame before rising edge
      # if line['v_ego'] < 10 * 0.447:
      faults.append(line)
    elif line['can']['lka_state'] == 5 and 25 not in [data[i + la + 1]['can']['lka_state'] for la in range(non_fault_look_ahead)]:  # gather samples where fault does not occur in x seconds
      non_faults.append(line)
del data

print('From data: {} faults, {} non-faults'.format(len(faults), len(non_faults)))

input_keys = ['torque_cmd', 'angle_steers', 'driver_torque', 'v_ego', 'angle_steers_des', 'angle_rate']

x_train = []
y_train = []
x_train_non_fault = []
y_train_non_fault = []

for line in faults:
  x_train.append([line[k] for k in input_keys])
  y_train.append(1)

for line in non_faults:
  x_train_non_fault.append([line[k] for k in input_keys])
  y_train_non_fault.append(0)

x_train_non_fault, _, y_train_non_fault, __ = train_test_split(x_train_non_fault, y_train_non_fault, test_size=0.9995)
x_train += x_train_non_fault
y_train += y_train_non_fault

print('Training on: {} faults, {} non faults'.format(y_train.count(1), y_train_non_fault.count(0)))

x_train = np.array(x_train)
y_train = np.array(y_train)


# Normalize to 0-1
scales = {k: [min(x_train.take(indices=idx, axis=1)), max(x_train.take(indices=idx, axis=1))] for idx, k in enumerate(input_keys)}
x_train = x_train.T
for idx, key in enumerate(input_keys):
  x_train[idx] = np.interp(x_train[idx], scales[key], [0, 1])
x_train = np.array(x_train).T

# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.01)


def norm(d, k):
  return np.interp(d, scales[k], [0, 1])


opt = Adam(amsgrad=True)
model = Sequential()
model.add(Dense(16, activation=LeakyReLU(), input_shape=x_train.shape[1:]))
model.add(Dense(16, activation=LeakyReLU()))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, y_train,
          # validation_data=(x_test, y_test),
          epochs=100, batch_size=16)


def fault_check(sample):
  return abs(sample['angle_rate']) > 100
  # return (sample['angle_steers'] < 0 < sample['angle_rate'] or sample['angle_steers'] > 0 > sample['angle_rate']) and abs(sample['angle_rate']) > 100
  return (sample['angle_rate'] * sample['can']['torque_cmd'] > 0 and abs(sample['angle_rate']) > 100) or ((sample['can']['driver_torque'] * sample['angle_rate'] > 0 and sample['angle_steers'] * sample['angle_rate'] < 0) and abs(sample['angle_rate']) > 100)
  # return sample['angle_rate'] * sample['can']['torque_cmd'] > 0 and abs(sample['can']['torque_cmd']) > 150 and abs(sample['angle_rate']) > 100  # this captures 71% of faults while only 0.99% of non faults
  return abs(sample['angle_steers']) < 100 and abs(sample['angle_rate']) > 100 and abs(sample['can']['driver_torque']) < 200


fault_results = list(map(fault_check, faults))
non_fault_results = list(map(fault_check, non_faults))

fault_results_model = list(model.predict([[np.interp(f[k], scales[k], [0, 1]) for k in input_keys] for f in faults]).reshape(-1) > 0.5)
non_fault_results_model = list(model.predict([[np.interp(f[k], scales[k], [0, 1]) for k in input_keys] for f in non_faults]).reshape(-1) > 0.5)

print('This check correctly caught {} of {} faults ({}%)'.format(fault_results.count(True), len(faults), round(fault_results.count(True) / len(faults) * 100, 2)))
print('This check incorrectly caught {} of {} non faults ({}%)'.format(non_fault_results.count(True), len(non_faults), round(non_fault_results.count(True) / len(non_faults) * 100, 2)))

print('Model check correctly caught {} of {} faults ({}%)'.format(fault_results_model.count(True), len(faults), round(fault_results_model.count(True) / len(faults) * 100, 2)))
print('Model check incorrectly caught {} of {} non faults ({}%)'.format(non_fault_results_model.count(True), len(non_faults), round(non_fault_results_model.count(True) / len(non_faults) * 100, 2)))

# lka_states = [line['can']['lka_state'] for sec in data for line in sec]

fig, ax = plt.subplots()
x = [f['angle_steers'] for f in faults]
y = [f['angle_rate'] for f in faults]


scale = [min([abs(f['can']['torque_cmd']) for f in faults]), max([abs(f['can']['torque_cmd']) for f in faults])]
z = [f['can']['torque_cmd'] for f in faults]

for i, txt in enumerate(z):
  # print(txt)
  ax.annotate(txt, (x[i] + 2, y[i]), size=9)

ax.scatter(x, y, c=np.interp(np.abs(z), scale, [0.9, 0.1]), cmap='gray')

x = [abs(f['angle_steers']) for f in non_faults]
y = [abs(f['angle_rate']) for f in non_faults]
# ax.scatter(x, y, c='red', s=1)


# ax.scatter(np.abs(z), np.abs(y))
plt.xlabel('angle')
plt.ylabel('rate')
# plt.ylim(0, 600)

# sns.distplot([f['angle_rate'] for f in faults], bins=27)
# sns.distplot([f['angle_steers'] for f in faults], bins=27)

# for l in [faults[idx] for idx, i in enumerate(fault_results) if not i]:
#   print(l)

# for l in [non_faults[idx] for idx, i in enumerate(non_fault_results) if i]:
#   print(l)
