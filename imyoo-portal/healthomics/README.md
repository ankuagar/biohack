# AWS HealthOmics clustering run

For the full click-by-click setup, see DETAILED_SETUP.md.

This package runs cluster_imyoo.py as a HealthOmics private WDL workflow. The H5AD is read from S3 and the
workflow exports the same four files that the Streamlit page expects:

- cluster_cells.parquet
- cluster_summary.parquet
- cluster_markers.parquet
- cluster_info.json

**Terminal map:** run Docker/ECR, bucket and IAM setup, workflow commands, status checks, and result downloads on the
local Mac. Run the organizer-bucket read and the 6 GB staging copy in the SageMaker terminal. HealthOmics reads the
staged S3 object, not the SageMaker filesystem.

The hackathon account is in us-east-1. The organizer source object is
s3://spc-biohack26-imyoo/flare-study/v3/imyoo_spc_biohackathon_20261017T211331Z.h5ad.
Stage it in the participant-owned bucket
s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/ before starting HealthOmics. Keep the input and
output S3 locations in the same AWS Region as the workflow. The HealthOmics run role must be allowed to read the
participant bucket, pull the ECR image, and write the output location.

## 0. Verify the account and input before creating anything

Run these commands in the SageMaker terminal. They do not download the 5.6 GB H5AD:

~~~bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export SOURCE_BUCKET=spc-biohack26-imyoo
export SOURCE_KEY=flare-study/v3/imyoo_spc_biohackathon_20261017T211331Z.h5ad

aws sts get-caller-identity --region "$AWS_REGION"
aws s3api head-object --bucket "$SOURCE_BUCKET" --key "$SOURCE_KEY" --region "$AWS_REGION"
~~~

The identity command should show the account you are allowed to use. The input check should return the object's
size and metadata. If either command fails, stop there and fix the AWS credentials or bucket access first.

## 0A. Stage the H5AD in a participant-owned bucket

Follow section 1A of `DETAILED_SETUP.md`. In summary, create
`imyoo-healthomics-241785579379-us-east-1` locally with AES256 default encryption and allow the SageMaker
`hackathon-data-reader` role to upload the input prefix. Then run this copy from the SageMaker terminal:

~~~bash
export AWS_REGION=us-east-1
export INPUT_BUCKET=imyoo-healthomics-241785579379-us-east-1

aws s3 cp \
  s3://spc-biohack26-imyoo/flare-study/v3/imyoo_spc_biohackathon_20261017T211331Z.h5ad \
  s3://${INPUT_BUCKET}/imyoo-clusters/v3/input/imyoo_v3.h5ad \
  --copy-props none \
  --sse AES256 \
  --region "$AWS_REGION"
~~~

A `head-object` from SageMaker can return 403 because that role has upload but not metadata-read access. Verify the
object from the local participant credentials.

## 1. Build and push the container

Run this from the repository root on a machine with Docker and AWS credentials:

~~~bash
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=imyoo-cluster

aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" || true
aws ecr get-login-password --region "$AWS_REGION" |
  docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
docker build -f healthomics/Dockerfile -t "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:1" .
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:1"
~~~

If Docker is unavailable in SageMaker, build and push the same Dockerfile from CloudShell, CodeBuild, or a local
machine. Do not put the H5AD or any patient data in the image.

After the push, verify that the image exists:

~~~bash
aws ecr describe-images --repository-name "$ECR_REPO" --region "$AWS_REGION"
~~~

HealthOmics also needs a repository policy. In the ECR console, open imyoo-cluster, choose Permissions, and add:

~~~json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "HealthOmicsPull",
    "Effect": "Allow",
    "Principal": {"Service": "omics.amazonaws.com"},
    "Action": [
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability"
    ],
    "Resource": "*"
  }]
}
~~~

## 2. Create the HealthOmics execution role

The role is different from the SageMaker user role. In IAM, choose Roles → Create role → Custom trust policy and
use this trust policy:

~~~json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "omics.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
~~~

Name it HealthOmicsImYooClusterRole. Add an inline policy with the following permissions, replacing
YOUR_ACCOUNT_ID if the identity command showed a different account:

~~~json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:AbortMultipartUpload",
        "s3:ListMultipartUploadParts"
      ],
      "Resource": "arn:aws:s3:::imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket", "s3:GetBucketLocation", "s3:ListBucketMultipartUploads"],
      "Resource": "arn:aws:s3:::imyoo-healthomics-241785579379-us-east-1"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:DescribeLogStreams",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:YOUR_ACCOUNT_ID:log-group:/aws/omics/WorkflowLog:log-stream:*"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup"],
      "Resource": "arn:aws:logs:us-east-1:YOUR_ACCOUNT_ID:log-group:/aws/omics/WorkflowLog:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchCheckLayerAvailability"
      ],
      "Resource": "arn:aws:ecr:us-east-1:YOUR_ACCOUNT_ID:repository/imyoo-cluster"
    }
  ]
}
~~~

If IAM refuses role creation or role passing, an AWS account administrator must create this role or use the
HealthOmics console option to create the required service role.

## 3. Create the private workflow

~~~bash
cd healthomics
zip -j imyoo-cluster-wdl.zip imyoo-cluster.wdl
aws omics create-workflow \
  --name "imyoo-single-cell-cluster" \
  --engine WDL \
  --definition-zip fileb://imyoo-cluster-wdl.zip \
  --storage-type DYNAMIC
~~~

Save the returned numeric workflow id. HealthOmics validates the WDL when the workflow is created.

## 4. Start a run

Copy run-parameters.example.json to run-parameters.json. The working parameters use the staged participant bucket:

~~~json
{
  "h5ad": "s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/input/imyoo_v3.h5ad",
  "container_image": "241785579379.dkr.ecr.us-east-1.amazonaws.com/imyoo-cluster:1",
  "resolution": 0.6,
  "neighbors": 15,
  "markers": 10,
  "max_cells": 100000
}
~~~

~~~bash
aws omics start-run \
  --workflow-id YOUR_WORKFLOW_ID \
  --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/HealthOmicsImYooClusterRole \
  --output-uri s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/ \
  --parameters file://run-parameters.json \
  --storage-type DYNAMIC \
  --name "imyoo-v3-clustering" \
  --retention-mode REMOVE
~~~

## 4A. Run a parameter study

Build image tag `2` after copying `cluster_study.py` into the image, create `imyoo-cluster-study.wdl` as a new
workflow, and use these study inputs:

~~~json
{
  "h5ad": "s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/input/imyoo_v3.h5ad",
  "container_image": "241785579379.dkr.ecr.us-east-1.amazonaws.com/imyoo-cluster:2",
  "resolutions": "0.2,0.4,0.6,0.8,1.0",
  "neighbors_list": "15",
  "max_cells": 100000
}
~~~

The study produces cluster summaries, configuration stability scores, and a common UMAP sample. To test neighbor
sensitivity separately, use `resolutions: "0.6"` and `neighbors_list: "10,15,30"`. Copy the four study outputs into
`data/` on the machine running Streamlit. For study run `YOUR_STUDY_RUN_ID`:

~~~bash
export STUDY_OUTPUT_URI=s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/study/YOUR_STUDY_RUN_ID
mkdir -p data
aws s3 cp "$STUDY_OUTPUT_URI/out/study_cells/cluster_study_cells.parquet" data/cluster_study_cells.parquet
aws s3 cp "$STUDY_OUTPUT_URI/out/study_summary/cluster_study_summary.parquet" data/cluster_study_summary.parquet
aws s3 cp "$STUDY_OUTPUT_URI/out/study_stability/cluster_study_stability.parquet" data/cluster_study_stability.parquet
aws s3 cp "$STUDY_OUTPUT_URI/out/study_info/cluster_study_info.json" data/cluster_study_info.json
~~~

The **🧪 Cluster studies** page will then be available.

## 5. Check status and get the results into the portal

Check the workflow status:

~~~bash
aws omics get-workflow \
  --id YOUR_WORKFLOW_ID \
  --region us-east-1 \
  --query '{id:id,status:status}' \
  --output table
~~~

Poll the run until the status is COMPLETED:

~~~bash
aws omics get-run \
  --id YOUR_RUN_ID \
  --region us-east-1 \
  --query '{status:status,message:statusMessage,start:startTime,stop:stopTime}' \
  --output table
~~~

If it reaches `FAILED`, inspect the `message` field. After `COMPLETED`, list the run output files:

~~~bash
aws s3 ls \
  s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/YOUR_RUN_ID/ \
  --recursive \
  --human-readable
~~~

The verified run places outputs under `out/<output-name>/`. Copy the four files into `data/` on the same machine that
runs the portal. For the live SageMaker portal, run these commands in the SageMaker terminal; for a local portal, run
them on the Mac:

~~~bash
export OUTPUT_URI=s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/9179041
mkdir -p data

aws s3 cp "$OUTPUT_URI/out/cluster_cells/cluster_cells.parquet" data/cluster_cells.parquet
aws s3 cp "$OUTPUT_URI/out/cluster_summary/cluster_summary.parquet" data/cluster_summary.parquet
aws s3 cp "$OUTPUT_URI/out/cluster_markers/cluster_markers.parquet" data/cluster_markers.parquet
aws s3 cp "$OUTPUT_URI/out/cluster_info/cluster_info.json" data/cluster_info.json
~~~

Restart or refresh Streamlit and open Unsupervised clusters.

## Resource choice

The WDL requests 8 vCPUs and 64 GB memory. HealthOmics maps that request to a memory-optimized omics.r.2xlarge
task in regions where it is available. If the run fails with an out-of-memory error, increase the task to 128 GB
and 16 vCPUs before rerunning.
