import pandas as pd

FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
            'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices']

def test_set():
    data_split = pd.read_csv("mimic-cxr-2.0.0-split.csv")

    test_set = []
    for i in range(len(data_split)):
        if data_split["split"][i] != "train":
            test_set.append("{}.{}".format(data_split["subject_id"][i], data_split["study_id"][i]))

    return test_set

def finding_distribution(test_set):
    data_chexpert = pd.read_csv("mimic-cxr-2.0.0-chexpert.csv")
    for f in FINDINGS:
        f_distribution = {"f_total":0,"f_positive":0,"f_uncertain":0}
        for i in range(len(data_chexpert[f])):
            if "{}.{}".format(data_chexpert["subject_id"][i], data_chexpert["study_id"][i]) not in test_set:
                f_distribution["f_total"] += 1
                if data_chexpert[f][i] == 1.0:
                    f_distribution["f_positive"] += 1
                elif data_chexpert[f][i] == -1.0:
                    f_distribution["f_uncertain"] += 1

        print(f)
        print(f_distribution)
        positive_incl = (f_distribution["f_positive"])/(f_distribution["f_total"])
        positive_excl = (f_distribution["f_positive"])/(f_distribution["f_total"]-f_distribution["f_uncertain"])
        print(round(positive_incl, 3)*100, "% [INCL U] ")
        print(round(positive_excl, 3)*100, "% [EXCL U] ")
        print()

def main():
    finding_distribution(test_set())

if __name__ == "__main__":
    main()

