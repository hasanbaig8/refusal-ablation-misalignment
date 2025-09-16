import os
import boto3
from botocore.config import Config
import dotenv
dotenv.load_dotenv()

def copy_models_to_s3(local_folder, s3_folder):
    # Configure S3 client for RunPod
    s3_client = boto3.client(
        's3',
        endpoint_url='https://s3api-eu-ro-1.runpod.io',
        aws_access_key_id=os.getenv('RUNPOD_AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('RUNPOD_AWS_SECRET_ACCESS_KEY'),
        region_name='eu-ro-1',
        config=Config(signature_version='s3v4')
    )
    
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

if __name__ == '__main__':
    copy_models_to_s3('models', 'refusal-ablation-misalignment/models')


