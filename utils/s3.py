import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_BUCKET = os.getenv('AWS_BUCKET')


class S3:
    bucket_name = AWS_BUCKET

    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )

    def list_buckets(self):
        return self.s3.list_buckets()

    def upload_audio(self, file_path, book_id, audio_id='audio'):
        ext = file_path.split('.')[-1]
        key = f'public/book/{book_id}/{audio_id}.{ext}'
        # mp3
        extra_args = {'ContentType': 'audio/mpeg'}
        self.s3.upload_file(file_path, self.bucket_name, key, ExtraArgs=extra_args)
        return f'https://{self.bucket_name}.s3.{AWS_REGION}.amazonaws.com/{key}', key

    def upload_pdf(self, file_path, book_id):
        ext = file_path.split('.')[-1].lower()
        key = f'public/book/{book_id}.{ext}'
        extra_args = {'ContentType': 'application/pdf'}
        self.s3.upload_file(file_path, self.bucket_name, key, ExtraArgs=extra_args)
        return f'https://{self.bucket_name}.s3.{AWS_REGION}.amazonaws.com/{key}', key

    def upload_image(self, file_path, book_id, image_id='image'):
        ext = file_path.split('.')[-1].lower()
        key = f'public/book/{book_id}/{image_id}.{ext}'
        if ext == 'png':
            content_type = 'image/png'
        else:
            content_type = 'image/jpeg'
        extra_args = {'ContentType': content_type}
        self.s3.upload_file(file_path, self.bucket_name, key, ExtraArgs=extra_args)
        return f'https://{self.bucket_name}.s3.{AWS_REGION}.amazonaws.com/{key}', key


if __name__ == '__main__':
    # Print out bucket names
    s3 = S3()
    print(s3.list_buckets())
