import sagemaker
from sagemaker.pytorch import PyTorch

# Define the S3 bucket and prefix
bucket = 'architectexam'
prefix = 'imgs'

# Define the SageMaker role
role = 'arn:aws:iam::511771194412:role/AmazonSageMaker-ExecutionRole'

# Create the PyTorch Estimator
pytorch_estimator = PyTorch(
    source_dir='code',
    entry_point='train.py',
    role=role,
    # instance_type='ml.p2.xlarge',
    instance_type='ml.m5.large',
    instance_count=1,
    framework_version='1.13.1',
    py_version='py39',
    # These will be picked up as arg parse
    # hyperparameters={
    #     'epochs': 10,
    #     'batch-size': 32,
    #     'learning-rate': 0.001,
    # }
)

# Start the training job
pytorch_estimator.fit({'training': f's3://{bucket}/{prefix}'})