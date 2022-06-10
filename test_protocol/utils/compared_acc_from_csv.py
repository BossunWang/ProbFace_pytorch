import csv
import os
import matplotlib.pyplot as plt
import numpy as np

log_files = os.listdir('../log')


dataset_list = set()
model_acc_dict = {}

for file in log_files:
    if file.endswith('.csv'):
        dataset_list.add(file.split('.csv')[0])
        file = os.path.join('../log', file)
        with open(file, mode='r') as infile:
            reader = csv.reader(infile)
            for rows in reader:
                if rows[0] != 'test set':
                    if rows[1] in model_acc_dict.keys():
                        model_acc_dict[rows[1]].append(float(rows[2]))
                    else:
                        model_acc_dict[rows[1]] = []
                        model_acc_dict[rows[1]].append(float(rows[2]))

reference_acc = np.array(model_acc_dict['SST_Prototype_Epoch_59_mask.pt'])
max_loss = -100.0

for key, values in model_acc_dict.items():
    if key == 'MLS_Epoch_9_mask_sst.pt' \
            or key == 'Backbone_IR_SE_101_Epoch_24_Time_2020-10-26-01-56_checkpoint.pt':
        continue

    loss = np.mean(np.array(model_acc_dict[key]) - reference_acc)

    if loss > max_loss:
        max_loss = loss
        max_loss_key = key

print('accurate over than SST_Prototype_Epoch_59_mask.pt:')
print(max_loss)
print(max_loss_key)
#
#     plt.plot(range(len(dataset_list)), values, label=key)
#     plt.xticks(range(len(dataset_list)), dataset_list)
# plt.legend()
# plt.show()