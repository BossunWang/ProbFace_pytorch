import csv
import os
import matplotlib.pyplot as plt
import numpy as np

log_files = os.listdir('../log_1N')


dataset_list = ['GEO_FR_Panel_1N_origin_crop']

key_map = {'SST_Prototype_Epoch_19_mask_magface_discface.pt': 'geo-mask_magface_discface'
           , 'Backbone_IR_SE_101_Epoch_24_Time_2020-10-26-01-56_checkpoint.pt': 'geo-001'
           , 'SST_Prototype_Epoch_19_nonmask_mask_magface_discface.pt': 'geo-nonmask_mask_magface_discface'
        #    , 'SST_Prototype_Epoch_19_mask_20210803.pt': 'geo-mask_20210803'
        #    , 'SST_Prototype_Epoch_19_mask_focal_loss_20210803.pt': 'geo-mask_focal_loss_20210803'
        #    , 'SST_Prototype_Epoch_19_nonmask_mask_20210803.pt': 'geo-nonmask_mask_20210803'
        #    , 'SST_Prototype_Epoch_19_nonmask_mask_focal_loss_20210803.pt': 'geo-nonmask_mask_focal_loss_20210803'
           , 'SST_Prototype_Epoch_29_ir100.pt': 'geo-nonmask_mask_ir100'
        #    , 'magface_epoch_00025_transfer.pt': 'magface'
           , 'FR_panel': 'FR_panel', 'FR_panel_geo001': 'FR_panel_geo001'}

for file in log_files:
    if file.endswith('.csv'):
        dataset_name = file.split('.csv')[0]
        file = os.path.join('../log_1N', file)

        model_dict = {}

        with open(file, mode='r') as infile:
            reader = csv.reader(infile)
            for rows in reader:
                if rows[0] != 'test set':
                    if rows[1] not in model_dict.keys():
                        model_dict[rows[1]] = []

                    model_dict[rows[1]].append(float(rows[2]))
                    model_dict[rows[1]].append(float(rows[4]))
                    model_dict[rows[1]].append(float(rows[3]))

        plt.figure(dataset_name)
        for key in model_dict.keys():
            if key not in key_map.keys():
                continue

            model_dict[key] = np.array(model_dict[key]).reshape(-1, 3)
            sort_index = np.argsort(model_dict[key][:, 0])
            model_dict[key] = model_dict[key][sort_index]
            threshold = list(model_dict[key][:, 0])
            plt.plot(model_dict[key][:, 1], model_dict[key][:, 2], '-o', label=key_map[key])

        print(threshold)

        plt.xlabel('FAR' + ' , t=' + ','.join([str(t) for t in threshold[::-1]]))
        plt.ylabel('TAR')
        plt.legend()
        plt.savefig(dataset_name + '.png')
        # plt.show()


# reference_acc = np.array(model_acc_dict['SST_Prototype_Epoch_59_mask.pt'])
# max_loss = -100.0
#
# for key, values in model_acc_dict.items():
#     if key == 'MLS_Epoch_9_mask_sst.pt' \
#             or key == 'Backbone_IR_SE_101_Epoch_24_Time_2020-10-26-01-56_checkpoint.pt':
#         continue
#
#     loss = np.mean(np.array(model_acc_dict[key]) - reference_acc)
#
#     if loss > max_loss:
#         max_loss = loss
#         max_loss_key = key
#
# print('accurate over than SST_Prototype_Epoch_59_mask.pt:')
# print(max_loss)
# print(max_loss_key)
# #
# #     plt.plot(range(len(dataset_list)), values, label=key)
# #     plt.xticks(range(len(dataset_list)), dataset_list)
# # plt.legend()
# # plt.show()