import boto3
import os
from botocore import UNSIGNED
from botocore.client import Config

# ----------------------------
# CONFIGURATION
# ----------------------------

LOCAL_BASE = '/Users/joemargolis/Desktop/ASD2/abide_sMRI_vbm'
BUCKET_NAME = 'fcp-indi'
PIPELINE = 'cpac'
STRATEGY = 'nofilt_noglobal'
DERIVATIVE = 'vbm'

S3_PREFIX = f"data/Projects/ABIDE_Initiative/Outputs/{PIPELINE}/{STRATEGY}/{DERIVATIVE}/"

# ----------------------------
# STEP 1 - Set up S3 client (anonymous)
# ----------------------------

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# ----------------------------
# STEP 2 - List all VBM files
# ----------------------------

print(f"Listing all VBM files under S3 prefix: {S3_PREFIX}")

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)

vbm_keys = []

for page in pages:
    if 'Contents' in page:
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('.nii.gz'):
                vbm_keys.append(key)

print(f"Found {len(vbm_keys)} VBM files in S3.")

# ----------------------------
# STEP 3 - Download all VBM files
# ----------------------------

if not os.path.exists(LOCAL_BASE):
    os.makedirs(LOCAL_BASE)

total_downloaded = 0

for key in vbm_keys:
    filename = os.path.basename(key)
    local_path = os.path.join(LOCAL_BASE, filename)

    if not os.path.exists(local_path):
        print(f"→ Downloading {key}")
        s3.download_file(BUCKET_NAME, key, local_path)
        total_downloaded += 1
    else:
        print(f"✓ Already downloaded: {key}")

print(f"All done!")
print(f"Total files downloaded: {total_downloaded}")

