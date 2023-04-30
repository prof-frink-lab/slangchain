ENVIRONMENT_NAME?=dev
REQUIRE_APPROVAL?=never
AWS_ACCOUNT_ID=$(shell aws sts get-caller-identity --query "Account" --output text)
AWS_ACCOUNT_REGION=$(shell aws configure get region)

bootstrap-local:
	AWS_ACCOUNT_ID=$(AWS_ACCOUNT_ID)  AWS_ACCOUNT_REGION=$(AWS_ACCOUNT_REGION) ENVIRONMENT_NAME=$(ENVIRONMENT_NAME) cdk bootstrap aws://$(AWS_ACCOUNT_ID)/$(AWS_ACCOUNT_REGION) --trust=$(AWS_ACCOUNT_ID) --cloudformation-execution-policies=arn:aws:iam::aws:policy/AdministratorAccess --verbose

destroy:
	AWS_ACCOUNT_ID=$(AWS_ACCOUNT_ID)  AWS_ACCOUNT_REGION=$(AWS_ACCOUNT_REGION)  ENVIRONMENT_NAME=$(ENVIRONMENT_NAME) cdk destroy --force

deploy-local:
	AWS_ACCOUNT_ID=$(AWS_ACCOUNT_ID) AWS_ACCOUNT_REGION=$(AWS_ACCOUNT_REGION) ENVIRONMENT_NAME=$(ENVIRONMENT_NAME) cdk deploy serverless-baby-agi  --require-approval $(REQUIRE_APPROVAL)

install: bootstrap-local deploy-local
