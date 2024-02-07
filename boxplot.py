import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv

def main():
    with open('/Users/yanggq/yanggq/Brats/prediction/brats_scores_2019.csv', newline='') as csvfile:
        # 创建一个csv.reader对象
        reader = csv.reader(csvfile)
        # 遍历每一行并输出
        #for row in reader:
        #    print(row)


        header = ("3D Unet", "BiTr-Unet", "HDC-Net", "TransBTS", "Ours")
        rows = list()
        subject_ids = list()
        for row in reader:
            # 将每行的第一列数据添加到列表中（假设您想要添加的是第一列数据）
            subject_ids.append(row[0])  # 此处假设您想要添加的是第一列数据
            rows.append([float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5])])
        print(subject_ids)
        print(rows)

        df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
        #df.to_csv("/Users/yanggq/yanggq/Brats/prediction/brats_scores11.csv")


        scores = dict()
        for index, score in enumerate(df.columns):
            values = df.values.T[index]
            scores[score] = values[np.isnan(values) == False]

        plt.boxplot(list(scores.values()), labels=list(scores.keys()))
        plt.ylabel("Dice Coefficient")
        plt.savefig("/Users/yanggq/yanggq/Brats/prediction/validation_scores_boxplot.png")
        plt.close()

if __name__ == "__main__":
    main()
