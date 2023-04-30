"""AWS CDK Stack script"""
import os
import typing
from typing import Dict, Any

import builtins

import constructs
from aws_cdk import (
  Stack,
  Duration,
  aws_iam as iam,
  aws_lambda as lambda_
)

import aws_cdk as cdk

app = cdk.App()

region_name = os.environ["AWS_ACCOUNT_REGION"]
deployment_environment = os.environ["ENVIRONMENT_NAME"]
app_name = app.node.try_get_context('configs').get('ApplicationName')
stack_name = f"{app_name}-{deployment_environment}"

class ServerlessAppStack(Stack):
  """Serverless App Stack Class"""

  def __init__(
    self,
    scope:typing.Optional[constructs.Construct],
    id:typing.Optional[builtins.str],
    **kwargs) -> None:
    """init"""
    super().__init__(scope, id, **kwargs)

    self.env_specific_variables : Dict[str, Any] = \
      self.node.try_get_context("configs").get(deployment_environment)

    self.create_lambda()

  def _create_lambda_role(self) -> iam.Role:
    """Create BabyAGI iam lambda function role"""

    iam_role = iam.Role(
      self,
      f"LambdaRole-{deployment_environment}",
      assumed_by = iam.ServicePrincipal("lambda.amazonaws.com"))

    iam_role.add_managed_policy( \
      iam.ManagedPolicy.from_aws_managed_policy_name(
        "service-role/AWSLambdaBasicExecutionRole"))

    return iam_role

  def create_lambda(self) -> lambda_.DockerImageFunction:
    """Create BabyAGI lambda function"""

    lambda_variables : Dict[str, Any] = \
      self.env_specific_variables.get("Lambda")

    docker_folder: str = lambda_variables.get("DockerFolder")
    filename: str = lambda_variables.get("FileName")
    handler: str = lambda_variables.get("HandlerFn")
    lambda_function_dir: str = lambda_variables.get("LambdaFunctionDir")
    log_level: str = lambda_variables.get("LogLevel")
    memory_mb: int = lambda_variables.get("MemoryMB")
    timeout_seconds: int = lambda_variables.get("TimeoutSeconds")
    max_concurrent_executions: int = lambda_variables.get("MaxConcurrentExecutions")
    openai_api_key: str = lambda_variables.get("OpenAIAPIKey")
    openai_model_name: str = lambda_variables.get("OpenAIModelName")
    openai_model_temperature: float = lambda_variables.get("OpenAIModelTemperature")
    search_results_num: int = lambda_variables.get("SearchResultsNum")
    wikipedia_results_num: int = lambda_variables.get("WikipediaResultsNum")

    max_concurrent_executions: str = lambda_variables.get("MaxConcurrentExecutions")

    iam_role = self._create_lambda_role()

    lambda_fn = lambda_.DockerImageFunction(
      self,
      f"LambdaFunction-{deployment_environment}",
      code=lambda_.DockerImageCode.from_image_asset(
        docker_folder,
        cmd=[ filename + f".{handler}" ],
        build_args={
          "FUNCTION_DIR": lambda_function_dir
        }),
      role=iam_role,
      memory_size=memory_mb,
      timeout=Duration.seconds(timeout_seconds),
      reserved_concurrent_executions=max_concurrent_executions,

      architecture=lambda_.Architecture.ARM_64 \
            if os.uname().machine == "arm64" else lambda_.Architecture.X86_64,
      environment={
        "NLTK_DATA": lambda_function_dir,
        "ENVIRONMENT": deployment_environment,
        "REGION_NAME": region_name,
        "LOG_LEVEL": log_level,
        "SEARCH_RESULTS_NUM": str(search_results_num),
        "WIKIPEDIA_RESULTS_NUM": str(wikipedia_results_num),
        "OPENAI_API_KEY": openai_api_key,
        "OPENAI_MODEL_NAME": openai_model_name,
        "LLM_TEMPERATURE": str(openai_model_temperature)
      }
    )

    return lambda_fn


lambda_stack = ServerlessAppStack(
  app,
  f"{app_name}",
  description="BabyAGI Serverless stack",
  stack_name=f"{app_name}-{deployment_environment}",
  env=cdk.Environment(
    account=os.environ["AWS_ACCOUNT_ID"],
    region=os.environ["AWS_ACCOUNT_REGION"]))

app.synth()
