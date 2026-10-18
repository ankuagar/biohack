# Detailed AWS HealthOmics setup for ImYoo clustering

This guide runs the ImYoo v3 single-cell clustering workflow in AWS HealthOmics and brings the results back into the
Streamlit portal.

The data locations from the AWS notes are:

- AWS Region: us-east-1
- ImYoo input:
  s3://spc-biohack26-imyoo/flare-study/v3/imyoo_spc_biohackathon_20261017T211331Z.h5ad
- Participant-owned bucket used for HealthOmics input and output:
  s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/

The workflow produces these files:

- cluster_cells.parquet
- cluster_summary.parquet
- cluster_markers.parquet
- cluster_info.json

## Terminal map

| Run on | Use it for |
|---|---|
| **Local Mac** | Docker/ECR, bucket creation and policies, IAM role updates, workflow creation, `start-run`, status checks, and downloading results. |
| **SageMaker terminal** | Reading the organizer-owned ImYoo source and copying the 6 GB H5AD into the participant-owned bucket. |

HealthOmics cannot read files that exist only on the SageMaker filesystem. The H5AD must be in the participant-owned
S3 bucket before `start-run` is executed from the Mac.

## 1. Verify AWS access

Run this in the SageMaker terminal. The commands only inspect metadata; they do not download the 5.6 GB H5AD:

~~~bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

export INPUT_BUCKET=spc-biohack26-imyoo
export INPUT_KEY=flare-study/v3/imyoo_spc_biohackathon_20261017T211331Z.h5ad

aws sts get-caller-identity --region "$AWS_REGION"
aws s3api head-object \
  --bucket "$INPUT_BUCKET" \
  --key "$INPUT_KEY" \
  --region "$AWS_REGION"
~~~

The second command should return the H5AD size and metadata. If it fails, fix AWS credentials or bucket access before
continuing.

## 1A. Create a participant-owned staging bucket

The organizer-owned ImYoo bucket is readable from SageMaker, but the HealthOmics role cannot use it directly. The
team scratch bucket also uses an organizer-owned KMS key. The working solution was a bucket in participant account
`241785579379` with S3-managed encryption (`AES256`). Create it from the local Mac using the BioHack participant
credentials:

~~~bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export INPUT_BUCKET=imyoo-healthomics-241785579379-us-east-1

aws s3api create-bucket --bucket "$INPUT_BUCKET" --region "$AWS_REGION"

aws s3api put-bucket-encryption \
  --bucket "$INPUT_BUCKET" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' \
  --region "$AWS_REGION"

aws s3api put-public-access-block \
  --bucket "$INPUT_BUCKET" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true \
  --region "$AWS_REGION"
~~~

Allow the SageMaker role to upload only to the input prefix. Run this policy command locally:

~~~bash
cat > /tmp/imyoo-bucket-policy.json <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SageMakerUpload",
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::241785579379:role/hackathon-data-reader"},
      "Action": ["s3:PutObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
      "Resource": "arn:aws:s3:::${INPUT_BUCKET}/imyoo-clusters/v3/input/*"
    },
    {
      "Sid": "SageMakerBucketAccess",
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::241785579379:role/hackathon-data-reader"},
      "Action": ["s3:GetBucketLocation", "s3:ListBucketMultipartUploads"],
      "Resource": "arn:aws:s3:::${INPUT_BUCKET}"
    },
    {
      "Sid": "SageMakerReadResults",
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::241785579379:role/hackathon-data-reader"},
      "Action": "s3:GetObject",
      "Resource": [
        "arn:aws:s3:::${INPUT_BUCKET}/imyoo-clusters/v3/9179041/out/*",
        "arn:aws:s3:::${INPUT_BUCKET}/imyoo-clusters/v3/study/*/out/*"
      ]
    }
  ]
}
JSON

aws s3api put-bucket-policy \
  --bucket "$INPUT_BUCKET" \
  --policy file:///tmp/imyoo-bucket-policy.json \
  --region "$AWS_REGION"
~~~

Now stage the 6 GB H5AD from the **SageMaker terminal**, not the Mac. The `--copy-props none` flag avoids the
source bucket's missing `s3:GetObjectTagging` permission, and `--sse AES256` creates an object the HealthOmics role
can read:

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

The copy command printed `copy: ...` and completed successfully. A `head-object` from SageMaker may return 403
because that role has upload permission but not read/metadata permission. Verify the object from the local participant
credentials instead:

~~~bash
aws s3api head-object \
  --bucket "$INPUT_BUCKET" \
  --key imyoo-clusters/v3/input/imyoo_v3.h5ad \
  --region "$AWS_REGION" \
  --query '{Size:ContentLength,Encryption:ServerSideEncryption}'
~~~

The expected result is approximately 6,001,929,836 bytes with `Encryption` equal to `AES256`.

### If you see ExpiredToken

The BioHackParticipant profile uses temporary access keys. Refresh them locally:

1. Open https://spc-biohack26.awsapps.com/start.
2. Open your team, choose BioHackParticipant, then choose Access keys.
3. Copy the Option 2 block.
4. Paste that block into the same local terminal where you will build and push the image. Do not paste keys into chat,
   source files, Dockerfiles, or Git.

The block should set all three temporary credentials:

~~~bash
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN
~~~

Then verify the refreshed credentials:

~~~bash
export AWS_REGION=us-east-1
export AWS_PROFILE=241785579379_BioHackParticipant
aws sts get-caller-identity --region "$AWS_REGION"
~~~

If you prefer to save the keys in the profile, edit the local file ~/.aws/credentials and replace the three key lines
under the single profile header [241785579379_BioHackParticipant]. Keep exactly one copy of that header. Do not run
aws sso login for this temporary-key profile; it has no SSO configuration.

## 2. Build and push the ECR container locally

Build this on your local computer. The build context must contain both:

- cluster_imyoo.py at its top level
- healthomics/Dockerfile

If your local checkout does not contain these files yet, copy them from SageMaker or update your checkout before
running Docker. Do not copy the H5AD into the build context.

The image must be Linux x86_64 because HealthOmics runs the task on x86_64 compute. The platform flag also makes this
work on an Apple Silicon Mac.

~~~bash
cd /path/to/imyoo-portal

export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=imyoo-cluster
export ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"

docker version
test -f cluster_imyoo.py
test -f healthomics/Dockerfile

aws ecr describe-repositories \
  --repository-names "$ECR_REPO" \
  --region "$AWS_REGION" >/dev/null 2>&1 || \
aws ecr create-repository \
  --repository-name "$ECR_REPO" \
  --region "$AWS_REGION"

aws ecr get-login-password --region "$AWS_REGION" |
  docker login \
    --username AWS \
    --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  -f healthomics/Dockerfile \
  -t "$ECR_URI:1" \
  --load .

docker push "$ECR_URI:1"

aws ecr describe-images \
  --repository-name "$ECR_REPO" \
  --region "$AWS_REGION"
~~~

If docker buildx is unavailable, use docker build --platform linux/amd64 -f healthomics/Dockerfile -t "$ECR_URI:1" .
Then run docker push "$ECR_URI:1".

If your AWS credentials are in a named profile, set it before the commands:

~~~bash
export AWS_PROFILE=241785579379_BioHackParticipant
~~~

## 3. Add the ECR repository policy

In the AWS Console:

1. Open Amazon ECR.
2. Open the imyoo-cluster repository.
3. Choose Permissions.
4. Edit the repository policy.
5. Add this policy:

~~~json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "HealthOmicsPull",
    "Effect": "Allow",
    "Principal": {
      "Service": "omics.amazonaws.com"
    },
    "Action": [
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability"
    ],
    "Resource": "*"
  }]
}
~~~

## 4. Create the HealthOmics execution role

This is separate from the SageMaker user role.

In IAM, choose Roles, Create role, and Custom trust policy. Use:

~~~json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Service": "omics.amazonaws.com"
    },
    "Action": "sts:AssumeRole"
  }]
}
~~~

Name the role:

~~~text
HealthOmicsImYooClusterRole
~~~

Add an inline policy with the following permissions. Replace YOUR_ACCOUNT_ID with the account returned by the
identity command. This policy uses the participant-owned bucket created in section 1A:

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

If IAM refuses role creation or role passing, an AWS account administrator must create the role or use the
HealthOmics console option to create the required service role.

The participant-owned bucket uses S3-managed AES256 encryption, so this workflow does not require access to the
organizer-owned scratch KMS key.

## 5. Create the HealthOmics workflow

~~~bash
cd ~/imyoo-portal/healthomics

zip -j imyoo-cluster-wdl.zip imyoo-cluster.wdl

aws omics create-workflow \
  --name "imyoo-single-cell-cluster" \
  --engine WDL \
  --definition-zip fileb://imyoo-cluster-wdl.zip \
  --storage-type DYNAMIC \
  --region us-east-1
~~~

Save the numeric workflow ID returned by the command.

## 6. Prepare the run parameters

From the healthomics directory:

~~~bash
cp run-parameters.example.json run-parameters.json
~~~

Set the file to:

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

Replace YOUR_ACCOUNT_ID with the account from aws sts get-caller-identity.

## 7. Start the run

~~~bash
aws omics start-run \
  --workflow-id YOUR_WORKFLOW_ID \
  --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/HealthOmicsImYooClusterRole \
  --output-uri s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/ \
  --parameters file://run-parameters.json \
  --storage-type DYNAMIC \
  --name "imyoo-v3-clustering" \
  --retention-mode REMOVE \
  --region us-east-1
~~~

Save the returned run ID.

The verified run used workflow `8048395` and returned run `9179041`. It completed successfully in about 24 minutes
(18:23:54 to 18:47:49 UTC), with output URI
`s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/9179041`. The run command was executed from the
local Mac; the 6 GB input copy was executed from SageMaker.

## 7A. Run a parameter study

The baseline workflow tests one configuration. `imyoo-cluster-study.wdl` runs a grid of Leiden configurations and
writes composition and stability tables. A configuration is the Cartesian product of `resolutions` and
`neighbors_list`. Start with a resolution sweep at 15 neighbors; add `10,15,30` later for neighbor sensitivity.

Build and push a new image tag from the **local Mac** because it includes `cluster_study.py`:

~~~bash
cd ~/imyoo-portal
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=241785579379
export ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/imyoo-cluster"

docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  -f healthomics/Dockerfile \
  -t "$ECR_URI:2" \
  --load .
docker push "$ECR_URI:2"
~~~

Create the study workflow locally:

~~~bash
cd ~/imyoo-portal/healthomics
zip -j imyoo-cluster-study-wdl.zip imyoo-cluster-study.wdl

aws omics create-workflow \
  --name "imyoo-cluster-parameter-study" \
  --engine WDL \
  --definition-zip fileb://imyoo-cluster-study-wdl.zip \
  --storage-type DYNAMIC \
  --region us-east-1
~~~

Save the new workflow ID, then create `run-parameters.study.json` locally. This first run compares five
resolutions on the 15-neighbor graph:

~~~json
{
  "h5ad": "s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/input/imyoo_v3.h5ad",
  "container_image": "241785579379.dkr.ecr.us-east-1.amazonaws.com/imyoo-cluster:2",
  "resolutions": "0.2,0.4,0.6,0.8,1.0",
  "neighbors_list": "15",
  "max_cells": 100000
}
~~~

Start it locally. Replace `YOUR_STUDY_WORKFLOW_ID`:

~~~bash
aws omics start-run \
  --workflow-id YOUR_STUDY_WORKFLOW_ID \
  --role-arn arn:aws:iam::241785579379:role/HealthOmicsImYooClusterRole \
  --output-uri s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/study/ \
  --parameters file://run-parameters.study.json \
  --storage-type DYNAMIC \
  --name "imyoo-cluster-resolution-study" \
  --retention-mode REMOVE \
  --region us-east-1
~~~

For a separate neighbor-sensitivity run, keep `resolutions` at `0.6` and set `neighbors_list` to `"10,15,30"`.
The output files are `cluster_study_cells.parquet`, `cluster_study_summary.parquet`,
`cluster_study_stability.parquet`, and `cluster_study_info.json`. Copy them into the portal's `data/` directory on
the machine running Streamlit. For a study run ID `YOUR_STUDY_RUN_ID`, the output copy commands are:

~~~bash
cd ~/imyoo-portal
export STUDY_OUTPUT_URI=s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/study/YOUR_STUDY_RUN_ID
mkdir -p data

aws s3 cp "$STUDY_OUTPUT_URI/out/study_cells/cluster_study_cells.parquet" data/cluster_study_cells.parquet
aws s3 cp "$STUDY_OUTPUT_URI/out/study_summary/cluster_study_summary.parquet" data/cluster_study_summary.parquet
aws s3 cp "$STUDY_OUTPUT_URI/out/study_stability/cluster_study_stability.parquet" data/cluster_study_stability.parquet
aws s3 cp "$STUDY_OUTPUT_URI/out/study_info/cluster_study_info.json" data/cluster_study_info.json
~~~

The portal will then show **🧪 Cluster studies**.

## 8. Monitor the run

Check that the workflow is ready before starting a new run:

~~~bash
aws omics get-workflow \
  --id YOUR_WORKFLOW_ID \
  --region us-east-1 \
  --query '{id:id,status:status}' \
  --output table
~~~

Poll the run. Replace `YOUR_RUN_ID` with the ID returned by `start-run` (the verified run was `9179041`):

~~~bash
aws omics get-run \
  --id YOUR_RUN_ID \
  --region us-east-1 \
  --query '{status:status,message:statusMessage,start:startTime,stop:stopTime}' \
  --output table
~~~

Wait for status `COMPLETED`. If it reaches `FAILED`, the `message` field is the first diagnostic to inspect.
After completion, list the files written to the run output prefix:

~~~bash
aws s3 ls \
  s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/YOUR_RUN_ID/ \
  --recursive \
  --human-readable
~~~

## 9. Copy results into the portal

The verified run placed outputs under `out/<output-name>/` and logs under `logs/`. Copy the four files into the portal
with these commands:

Run the following on the machine that runs the portal. If the live portal is in SageMaker, run them in the SageMaker
terminal so the files land in `~/imyoo-portal/data/`. If you are running the portal locally, run them on the Mac.
Copy each URI into the matching local file:

~~~bash
cd ~/imyoo-portal

export OUTPUT_URI=s3://imyoo-healthomics-241785579379-us-east-1/imyoo-clusters/v3/9179041
mkdir -p data

aws s3 cp "$OUTPUT_URI/out/cluster_cells/cluster_cells.parquet" data/cluster_cells.parquet
aws s3 cp "$OUTPUT_URI/out/cluster_summary/cluster_summary.parquet" data/cluster_summary.parquet
aws s3 cp "$OUTPUT_URI/out/cluster_markers/cluster_markers.parquet" data/cluster_markers.parquet
aws s3 cp "$OUTPUT_URI/out/cluster_info/cluster_info.json" data/cluster_info.json
~~~

Refresh or restart Streamlit and open Unsupervised clusters.

## 10. Expected resource use

The WDL requests 8 vCPUs and 64 GB memory. HealthOmics maps that request to a memory-optimized omics.r.2xlarge task
where available. If the run fails with out-of-memory, increase the WDL task to 16 vCPUs and 128 GB memory before
rerunning.
