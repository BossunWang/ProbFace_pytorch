""" 
@author: Jixuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com
"""

import scipy.io as scio
from abc import ABCMeta, abstractmethod
import numpy as np
import os
import natsort


class PairsParser(metaclass=ABCMeta):
    """Parse the pair list for lfw based protocol.
    Because the official pair list for different dataset(lfw, cplfw, calfw ...) is different, 
    we need different method to parse the pair list of different dataset.

    Attributes:
        croped_face_folder(str): the root directory of test set for p1.
        croped_face_other_folder(str): the root directory of test set for p2.
        pairs_file(str): the path of the pairs file that was released by official.
    """
    def __init__(self, croped_face_folder, croped_face_other_folder, pairs_file):
        """Init PairsParser
            
        Args:
            croped_face_folder(str): the root directory of test set for p1.
            croped_face_other_folder(str): the root directory of test set for p2.
            pairs_file(str): the path of the pairs file that was released by official.
        """
        self.croped_face_folder = croped_face_folder
        self.croped_face_other_folder = croped_face_other_folder
        self.pairs_file = pairs_file

    def parse_pairs(self):
        """The method for parsing pair list.
        """
        pass


class LFW_PairsParser(PairsParser):
    """The pairs parser for lfw.
    """
    def parse_pairs(self, postponement='.jpg'):
        test_pair_list = []
        dirFiles = os.listdir(self.croped_face_folder)  # list of directory files
        dirFiles = natsort.natsorted(dirFiles)
        labels = np.load(self.pairs_file)

        for i, label in enumerate(labels):
            p1_name = str(i) + '_p1' + postponement
            p2_name = str(i) + '_p2' + postponement
            if p1_name in dirFiles and p2_name in dirFiles:
                p1_index = dirFiles.index(p1_name)
                p2_index = dirFiles.index(p2_name)
                test_pair_list.append((dirFiles[p1_index], dirFiles[p2_index], label))

        return test_pair_list


class RFW_PairsParser(PairsParser):
    """The pairs parser for rfw.
    """
    def parse_pairs(self):
        test_pair_list = []
        pairs_file_buf = open(self.pairs_file)
        line = pairs_file_buf.readline() # skip first line
        line = pairs_file_buf.readline().strip()
        while line:
            line_strs = line.split('\t')
            if len(line_strs) == 3:
                person_name = line_strs[0]
                image_index1 = line_strs[1]
                image_index2 = line_strs[2]
                image_name1 = person_name + '/' + person_name + '_' + image_index1.zfill(4) + '.jpg'
                image_name2 = person_name + '/' + person_name + '_' + image_index2.zfill(4) + '.jpg'
                label = 1
            elif len(line_strs) == 4:
                person_name1 = line_strs[0]
                image_index1 = line_strs[1]
                person_name2 = line_strs[2]
                image_index2 = line_strs[3]
                image_name1 = person_name1 + '/' + person_name1 + '_' + image_index1.zfill(4) + '.jpg'
                image_name2 = person_name2 + '/' + person_name2 + '_' + image_index2.zfill(4) + '.jpg'
                label = 0
            else:
                raise Exception('Line error: %s.' % line)

            image_path1 = os.path.join(self.croped_face_folder, image_name1)
            image_path2 = os.path.join(self.croped_face_other_folder, image_name2)

            if os.path.isfile(image_path1) and os.path.isfile(image_path2):
                test_pair_list.append((image_name1, image_name2, label))
            line = pairs_file_buf.readline().strip()

        return test_pair_list


class CPLFW_PairsParser(PairsParser):
    """The pairs parser for cplfw.
    """
    def parse_pairs(self):
        test_pair_list = []
        dirFiles = os.listdir(self.croped_face_folder)  # list of directory files
        dirFiles = natsort.natsorted(dirFiles)
        labels = np.load(self.pairs_file)

        for i, label in enumerate(labels):
            p1_name = str(i) + '_p1.jpg'
            p2_name = str(i) + '_p2.jpg'
            if p1_name in dirFiles and p2_name in dirFiles:
                p1_index = dirFiles.index(p1_name)
                p2_index = dirFiles.index(p2_name)
                test_pair_list.append((dirFiles[p1_index], dirFiles[p2_index], label))

        return test_pair_list


class CALFW_PairsParser(PairsParser):
    """The pairs parser for calfw.
    """
    def parse_pairs(self):
        test_pair_list = []
        dirFiles = os.listdir(self.croped_face_folder)  # list of directory files
        dirFiles = natsort.natsorted(dirFiles)
        labels = np.load(self.pairs_file)

        for i, label in enumerate(labels):
            p1_name = str(i) + '_p1.jpg'
            p2_name = str(i) + '_p2.jpg'
            if p1_name in dirFiles and p2_name in dirFiles:
                p1_index = dirFiles.index(p1_name)
                p2_index = dirFiles.index(p2_name)
                test_pair_list.append((dirFiles[p1_index], dirFiles[p2_index], label))

        return test_pair_list


class AgeDB_PairsParser(PairsParser):
    """The pairs parser for agedb.
    """
    def parse_pairs(self):
        test_pair_list = []
        dirFiles = os.listdir(self.croped_face_folder)  # list of directory files
        dirFiles = natsort.natsorted(dirFiles)
        labels = np.load(self.pairs_file)

        for i, label in enumerate(labels):
            p1_name = str(i) + '_p1.jpg'
            p2_name = str(i) + '_p2.jpg'
            if p1_name in dirFiles and p2_name in dirFiles:
                p1_index = dirFiles.index(p1_name)
                p2_index = dirFiles.index(p2_name)
                test_pair_list.append((dirFiles[p1_index], dirFiles[p2_index], label))

        return test_pair_list


class CFP_FP_PairsParser(PairsParser):
    """The pairs parser for cfp_fp.
    """
    def parse_pairs(self):
        test_pair_list = []
        dirFiles = os.listdir(self.croped_face_folder)  # list of directory files
        dirFiles = natsort.natsorted(dirFiles)
        labels = np.load(self.pairs_file)

        for i, label in enumerate(labels):
            p1_name = str(i) + '_p1.jpg'
            p2_name = str(i) + '_p2.jpg'
            if p1_name in dirFiles and p2_name in dirFiles:
                p1_index = dirFiles.index(p1_name)
                p2_index = dirFiles.index(p2_name)
                test_pair_list.append((dirFiles[p1_index], dirFiles[p2_index], label))

        return test_pair_list


class Masked_whn_PairsParser(PairsParser):
    """The pairs parser for agedb.
    """
    def parse_pairs(self):
        test_pair_list = []
        pair_list = open(self.pairs_file, "r")
        pair_list = [line.rstrip() for line in pair_list]

        for pair_line in pair_list:
            pair_line_list = pair_line.split(' ')
            filepath1 = os.path.join(self.croped_face_folder, pair_line_list[0])
            filepath2 = os.path.join(self.croped_face_folder, pair_line_list[1])

            if os.path.isfile(filepath1) and os.path.isfile(filepath2):
                label = 1 if pair_line_list[2] == '1' else 0
                test_pair_list.append((pair_line_list[0], pair_line_list[1], label))

        return test_pair_list


class MLFW_PairsParser(PairsParser):
    """The pairs parser for agedb.
    """
    def parse_pairs(self):
        test_pair_list = []
        pair_list = open(self.pairs_file, "r")
        pair_list = [line.rstrip() for line in pair_list]

        for pair_line in pair_list:
            pair_line_list = pair_line.split('	')
            filepath1 = os.path.join(self.croped_face_folder, pair_line_list[0])
            filepath2 = os.path.join(self.croped_face_folder, pair_line_list[1])

            if os.path.isfile(filepath1) and os.path.isfile(filepath2):
                label = 1 if pair_line_list[2] == '1' else 0
                test_pair_list.append((pair_line_list[0], pair_line_list[1], label))

        return test_pair_list


class NIST_SD32_MEDS_II_PairsParser(PairsParser):
    """The pairs parser for agedb.
    """
    def parse_pairs(self, postponement='.jpg'):
        test_pair_list = []
        pair_list = open(self.pairs_file, "r")
        pair_list = [line.rstrip() for line in pair_list]

        for pair_line in pair_list:
            pair_line_list = pair_line.split(' ')
            pair_line1 = pair_line_list[0].split('.jpg')[0] + postponement
            pair_line2 = pair_line_list[1].split('.jpg')[0] + postponement
            filepath1 = os.path.join(self.croped_face_folder, pair_line1)
            filepath2 = os.path.join(self.croped_face_other_folder, pair_line2)

            if os.path.isfile(filepath1) and os.path.isfile(filepath2):
                label = 1 if pair_line_list[2] == '1' else 0
                test_pair_list.append((pair_line1, pair_line2, label))

        return test_pair_list


class PairsParserFactory(object):
    """The factory used to produce different pairs parser for different dataset.

    Attributes:
        pairs_file(str): the path of the pairs file that was released by official.
        test_set(str): the name of different dataset.
    """
    def __init__(self, croped_face_folder, croped_face_other_folder, pairs_file, test_set):
        self.croped_face_folder = croped_face_folder
        self.croped_face_other_folder = croped_face_other_folder
        self.pairs_file = pairs_file
        self.test_set = test_set

    def get_parser(self):
        if self.test_set == 'CPLFW' or ('CPLFW_MASK' in self.test_set):
            pairs_parser = CPLFW_PairsParser(self.croped_face_folder, self.croped_face_other_folder, self.pairs_file)
        elif self.test_set == 'CALFW' or ('CALFW_MASK' in self.test_set):
            pairs_parser = CALFW_PairsParser(self.croped_face_folder, self.croped_face_other_folder, self.pairs_file)
        elif 'RFW_' in self.test_set:
            pairs_parser = RFW_PairsParser(self.croped_face_folder, self.croped_face_other_folder, self.pairs_file)
        elif self.test_set == 'LFW' or ('LFW_MASK' in self.test_set):
            pairs_parser = LFW_PairsParser(self.croped_face_folder, self.croped_face_other_folder, self.pairs_file)
        elif self.test_set == 'AgeDB30' or ('AgeDB30_MASK' in self.test_set):
            pairs_parser = AgeDB_PairsParser(self.croped_face_folder, self.croped_face_other_folder, self.pairs_file)
        elif self.test_set == 'CFP_FP' or ('CFP_FP_MASK' in self.test_set):
            pairs_parser = CFP_FP_PairsParser(self.croped_face_folder, self.croped_face_other_folder, self.pairs_file)
        elif self.test_set == 'Masked_whn':
            pairs_parser = Masked_whn_PairsParser(self.croped_face_folder, self.croped_face_other_folder, self.pairs_file)
        elif self.test_set == 'MLFW':
            pairs_parser = MLFW_PairsParser(self.croped_face_folder, self.croped_face_other_folder, self.pairs_file)
        elif 'NIST_SD32_MEDS_II' in self.test_set:
            pairs_parser = NIST_SD32_MEDS_II_PairsParser(self.croped_face_folder, self.croped_face_other_folder, self.pairs_file)
        else:
            pairs_parser = None
        return pairs_parser


if __name__ == '__main__':
    croped_face_folder1 = '/home/bossun/face_dataset/NIST_SD32_MEDS_II_face_ppm'
    croped_face_folder2 = '/home/bossun/face_dataset/NIST_SD32_MEDS_II_face_FMA_mask_ppm'
    pairs_file = '/home/bossun/face_dataset/NIST_SD32_MEDS_II_face_FMA_mask_arcface_pairs.txt'
    pairs_parser = NIST_SD32_MEDS_II_PairsParser(croped_face_folder1, croped_face_folder2, pairs_file)
    test_pair_list = pairs_parser.parse_pairs(postponement='.ppm')
    print(test_pair_list)

    NIST_SD32_MEDS_II_path = "NIST_SD32_MEDS_II_list.txt"
    f = open(NIST_SD32_MEDS_II_path, 'w')

    labels_list = []
    index_count = 1
    for test_pair in test_pair_list:
        test_pair1 = os.path.join(croped_face_folder1.split('/')[-1], test_pair[0])
        test_pair2 = os.path.join(croped_face_folder2.split('/')[-1], test_pair[1])
        f.write(str(index_count) + ' ' + test_pair1 + ' MUGSHOT\n')
        f.write(str(index_count + 1) + ' ' + test_pair2 + ' MUGSHOT\n')

        index_count += 2

        label = bool(test_pair[2])
        labels_list.append(label)

    f.close()

    np.save('NIST_SD32_MEDS_II_list.npy', labels_list)

    labels_list = np.load('NIST_SD32_MEDS_II_list.npy')
    print('labels_list:', labels_list)