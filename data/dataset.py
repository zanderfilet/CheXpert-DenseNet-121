import pandas as pd
import numpy as np
import random, csv
from tqdm import tqdm

FILEPATH = "files"
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
            'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices']

FINDINGS_DISTRIBUTION = {'Atelectasis': 0.210, 'Cardiomegaly': 0.201, 'Consolidation': 0.048, 'Edema': 0.124,
                         'Enlarged Cardiomediastinum': 0.033, 'Fracture': 0.019, 'Lung Lesion': 0.027,
                         'Lung Opacity': 0.23, 'No Finding': 0.333, 'Pleural Effusion': 0.243, 'Pleural Other': 0.009,
                         'Pneumonia': 0.079, 'Pneumothorax': 0.046, 'Support Devices': 0.292} # /utils/finding_distribution.py

#LSR Ranges u~(0, 100)
LSR_Ones = [55, 85]
LSR_Zeros = [0, 30]

def main():
    #Go through each PATIENT > STUDY --> Filter out
    #If study in

    data_meta = pd.read_csv("mimic-cxr-2.0.0-metadata.csv")
    data_chexpert = pd.read_csv("mimic-cxr-2.0.0-chexpert.csv")
    data_split = pd.read_csv("mimic-cxr-2.0.0-split.csv")

    #print([x for x in data_meta]) >> ['dicom_id', 'subject_id', 'study_id', 'PerformedProcedureStepDescription', 'ViewPosition', 'Rows', 'Columns', 'StudyDate', 'StudyTime', 'ProcedureCodeSequence_CodeMeaning', 'ViewCodeSequence_CodeMeaning', 'PatientOrientationCodeSequence_CodeMeaning']
    #print([x for x in data_chexpert]) >> ['subject_id', 'study_id', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    #print([x for x in data_split]) >> ['dicom_id', 'study_id', 'subject_id', 'split']

    print("[PATHS]")

    path_dir = {}
    for i in tqdm(range(len(data_meta))):
        if data_meta["ViewPosition"][i] != "LATERAL":
            subject_id = data_meta["subject_id"][i]
            study_id = data_meta["study_id"][i]
            dicom_id = data_meta["dicom_id"][i]
            path = "/{}/p{}/p{}/s{}/{}.jpg".format(FILEPATH, str(subject_id)[0:2], subject_id, study_id, dicom_id)
            path_dir["{}.{}".format(subject_id,study_id)] = path

    split_dir = {}

    print("[FINDINGS]")

    for i in tqdm(range(len(data_chexpert))):
        subject_id = data_chexpert["subject_id"][i]
        study_id = data_chexpert["study_id"][i]

        try: # cancel missing entries in keys
            findings_results = [path_dir["{}.{}".format(subject_id, study_id)]]
            for f in FINDINGS:
                f_float = data_chexpert[f].replace(np.nan, 0)[i]

                if f_float == -1.0:  # U-MultiClass+LSR

                    U_Finding = np.random.choice([1, 0], 1, p=[FINDINGS_DISTRIBUTION[f], 1 - FINDINGS_DISTRIBUTION[f]])

                    if U_Finding == 1:  # LSR_U-Ones
                        f_float = random.randint(LSR_Ones[0], LSR_Ones[1]) / 100

                    else:  # LSR_U-Zeros
                        f_float = random.randint(LSR_Zeros[0], LSR_Zeros[1]) / 100

                findings_results.append(f_float)

            split_dir["{}.{}".format(subject_id, study_id)] = findings_results
        except:
            pass



    print("[SPLIT]")

    categories = {"train": [], "validate": [], "test": []}

    p_ssid = "xxx"
    for i in tqdm(range(len(data_split))):
        try:
            subject_study_id = "{}.{}".format(data_split["subject_id"][i], data_split["study_id"][i])
            if subject_study_id != p_ssid:
                entry = split_dir[subject_study_id]
                categories[data_split["split"][i]].append(entry)
            p_ssid = subject_study_id
        except:
            pass

    for category in categories.items():
        with open('./config/{}.csv'.format(category[0]), 'w') as f:
            fieldnames = ['Path', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
             'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
             'Pneumothorax', 'Support Devices']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            for entry in category[1]:
                #print(entry)
                w.writerow(dict(zip(fieldnames, entry)))


if __name__ == "__main__":
    main()

