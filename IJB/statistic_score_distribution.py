import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def read_template_pair_list(path):
    # pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    # print(pairs.shape)
    # print(pairs[:, 0].astype(np.int))
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


def main():
    result_dir = "ir101_adaface_uc"
    target = "IJBC"
    job = "ir101_adaface_uc"
    save_path = result_dir + '/%s_result' % target
    score_save_file = os.path.join(save_path, "%s.npy" % job)
    score = np.load(score_save_file)

    mean_score = np.mean(score)
    min_score = np.min(score)
    max_score = np.max(score)
    print("min: {}, max: {}".format(min_score, max_score))

    image_path = "/media/glory/Transcend/Dataset/Face_Recognition_Dataset/IJB-C/IJB/IJB_release/IJBC"
    p1, p2, label = read_template_pair_list(
        os.path.join('%s/meta' % image_path, '%s_template_pair_label.txt' % target.lower()))

    # plot
    plot_img_filename = os.path.join(target + '_' + job + 'statistic.png')
    plt.figure(target + '_' + job)
    sns.distplot(score[label == 1], label="G")
    sns.distplot(score[label == 0], label="I")
    plt.axvline(mean_score, 0, 1)
    plt.legend()
    plt.savefig(plot_img_filename)
    plt.close("all")


if __name__ == '__main__':
    main()