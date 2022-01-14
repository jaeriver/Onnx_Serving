docker build -t onnx_lambda_container . --no-cache

export ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)

docker tag onnx_lambda_container $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/onnx_lambda_container

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com

docker push $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/onnx_lambda_container
