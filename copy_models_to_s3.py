import os
import boto3
from botocore.config import Config
import dotenv
dotenv.load_dotenv()

def get_s3_client():
    """Get configured S3 client for RunPod"""
    return boto3.client(
        's3',
        endpoint_url='https://s3api-eu-ro-1.runpod.io',
        aws_access_key_id=os.getenv('RUNPOD_AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('RUNPOD_AWS_SECRET_ACCESS_KEY'),
        region_name='eu-ro-1',
        config=Config(signature_version='s3v4')
    )

def copy_models_to_s3(local_folder, s3_folder):
    s3_client = get_s3_client()
    bucket_name = 'pfaieww9xc'
    
    if not os.path.exists(local_folder):
        print(f"Directory {local_folder} does not exist")
        return
    
    # Walk through the local directory and upload all files
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Create S3 key by preserving the directory structure relative to the local folder
            relative_path = os.path.relpath(local_file_path, local_folder)
            s3_key = os.path.join(s3_folder, relative_path).replace('\\', '/')  # Ensure forward slashes for S3
            
            try:
                print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
                s3_client.upload_file(local_file_path, bucket_name, s3_key)
                print(f"Successfully uploaded {s3_key}")
            except Exception as e:
                print(f"Error uploading {local_file_path}: {str(e)}")

def copy_models_from_s3(s3_folder, local_folder):
    s3_client = get_s3_client()
    bucket_name = 'pfaieww9xc'
    
    # Create local directory if it doesn't exist
    os.makedirs(local_folder, exist_ok=True)
    
    try:
        # List all objects in the S3 folder
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)
        
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # Skip directories (keys ending with '/')
                    if s3_key.endswith('/'):
                        continue
                    
                    # Create local file path by removing the s3_folder prefix
                    relative_path = os.path.relpath(s3_key, s3_folder)
                    local_file_path = os.path.join(local_folder, relative_path)
                    
                    # Create local directory structure if needed
                    local_dir = os.path.dirname(local_file_path)
                    os.makedirs(local_dir, exist_ok=True)
                    
                    try:
                        print(f"Downloading s3://{bucket_name}/{s3_key} to {local_file_path}")
                        s3_client.download_file(bucket_name, s3_key, local_file_path)
                        print(f"Successfully downloaded {s3_key}")
                    except Exception as e:
                        print(f"Error downloading {s3_key}: {str(e)}")
                        
    except Exception as e:
        print(f"Error listing objects in s3://{bucket_name}/{s3_folder}: {str(e)}")

if __name__ == '__main__':
    copy_models_to_s3('models', 'refusal-ablation-misalignment/models')
    copy_models_from_s3('refusal-ablation-misalignment/models', 'models')