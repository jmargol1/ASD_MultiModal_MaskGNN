import boto3
import os
import pandas as pd

# ----------------------------
# CONFIGURATION
# ----------------------------

LOCAL_BASE = '/Users/joemargolis/Desktop/ASD2/abide_sMRI_freesurfer'
PHENO_FILE = 'Phenotypic_V1_0b_preprocessed1.csv'
AGE_THRESHOLD = 13
BUCKET_NAME = 'fcp-indi'
S3_PREFIX_BASE = 'data/Projects/ABIDE_Initiative/Outputs/freesurfer'

# ----------------------------
# STEP 1 - Load phenotypic file
# ----------------------------

print(f"Loading phenotypic file: {PHENO_FILE}")
pheno = pd.read_csv(PHENO_FILE)

# Filter subjects under age 13
pheno_filtered = pheno[pheno['AGE_AT_SCAN'] < AGE_THRESHOLD]

# Count ASD vs. TDC
n_total = len(pheno_filtered)
n_asd = len(pheno_filtered[pheno_filtered['DX_GROUP'] == 1])
n_tdc = len(pheno_filtered[pheno_filtered['DX_GROUP'] == 2])

print(f"Total subjects under age {AGE_THRESHOLD}: {n_total}")
print(f"   ASD: {n_asd}")
print(f"   TDC: {n_tdc}")

if n_total == 0:
    print("No subjects meet criteria. Exiting.")
    exit()

# List of unique subject IDs under 13
subject_ids = pheno_filtered['FILE_ID'].dropna().unique().tolist()

# ----------------------------
# STEP 2 - Setup S3 client
# ----------------------------

s3 = boto3.client('s3')

# ----------------------------
# STEP 3 - Download Freesurfer folders
# ----------------------------

for subj_id in subject_ids:
    s3_prefix = f"{S3_PREFIX_BASE}/{subj_id}/"

    # Check if Freesurfer folder exists
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=s3_prefix, MaxKeys=1)
    if 'Contents' not in response:
        print(f"Skipping {subj_id} → No Freesurfer data found.")
        continue

    print(f"Downloading Freesurfer data for {subj_id}...")

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix)

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                rel_path = key[len(S3_PREFIX_BASE)+1:]
                local_path = os.path.join(LOCAL_BASE, rel_path)

                local_dir = os.path.dirname(local_path)
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)

                if not os.path.exists(local_path):
                    print(f"→ Downloading {key}")
                    s3.download_file(BUCKET_NAME, key, local_path)
                else:
                    print(f"✓ Already downloaded: {key}")

print("All done!")

