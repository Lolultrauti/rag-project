# Deploy Helix RAG to AWS (EC2 + Docker Compose)

One EC2 instance runs the API and a pgvector Postgres together via the project's
Docker Compose. CloudFormation provisions everything. The Gemini API key is stored
in SSM Parameter Store and read at boot — it is never written into the template or
EC2 user-data.

## Cost (rough)

- `t3.small` on-demand ≈ **$0.02/hr (~$15/mo)** if left running; 30 GB gp3 ≈ $2.40/mo.
- New AWS accounts: t3.micro is free-tier, but use t3.small here (build needs ~2 GB RAM).
- **Stop the instance when idle** to pay only for storage.

## Prerequisites

1. AWS account + the **AWS CLI** installed and configured (`aws configure` with an
   IAM user/role that can create EC2, IAM, and CloudFormation resources).
2. A **billing-enabled Gemini API key** (free tier = ~20 answers/day — unusable for real traffic).
3. Pick a region, e.g. `us-east-1`.

## Step 1 — store the Gemini key in SSM (one time)

```bash
aws ssm put-parameter \
  --name /helix/gemini_api_key \
  --type SecureString \
  --value "YOUR_REAL_GEMINI_KEY" \
  --region us-east-1
```

## Step 2 — deploy the stack

```bash
aws cloudformation deploy \
  --stack-name helix-rag \
  --template-file deploy/aws/cloudformation.yaml \
  --capabilities CAPABILITY_IAM \
  --region us-east-1
# Optional overrides:
#   --parameter-overrides InstanceType=t3.medium AppPortCidr=YOUR.IP.ADD.R/32
```

The instance boots, installs Docker, clones this repo, and runs `docker compose up
-d --build`. First build takes **~3–5 minutes** after the stack reports CREATE_COMPLETE.

## Step 3 — open the app

```bash
aws cloudformation describe-stacks --stack-name helix-rag \
  --query "Stacks[0].Outputs" --output table --region us-east-1
```

Open the `AppURL` (e.g. `http://ec2-x-x-x-x.compute-1.amazonaws.com:8000`). The UI is
at `/`, API docs at `/docs`.

If it's not up yet, the image is still building — wait a minute. To watch it:
**Systems Manager → Session Manager → Start session** on the `helix-rag` instance
(no SSH key needed), then:

```bash
sudo cat /var/log/cloud-init-output.log     # bootstrap progress
cd /opt/helix && sudo docker compose ps      # container status
sudo docker compose logs app | tail -50
```

## Updating after you push new code

```bash
# via Session Manager on the instance:
cd /opt/helix
sudo git pull
sudo docker compose up -d --build
```

## Tear down (stop all charges)

```bash
aws cloudformation delete-stack --stack-name helix-rag --region us-east-1
aws ssm delete-parameter --name /helix/gemini_api_key --region us-east-1   # if done with it
```

## Hardening (when you go past a demo)

- **HTTPS:** put a reverse proxy with a real domain in front of port 8000 — add a
  Caddy or Nginx container (Caddy auto-provisions Let's Encrypt certs), and restrict
  the security group so only the proxy is public.
- **Managed database:** move Postgres to **RDS** (supports `pgvector`) so data survives
  instance loss and gets automated backups; point `DATABASE_URL` at the RDS endpoint
  and drop the `db` service from compose.
- **Narrow access:** set `AppPortCidr` to your IP while testing.
- **Secrets:** already done right (SSM SecureString). For rotation, use Secrets Manager.
- **Backups:** snapshot the EBS volume, or rely on RDS automated backups once migrated.

## Why this shape (not ECS/Fargate)

For a portfolio/demo, one EC2 box reusing the exact Docker Compose we already
build-and-run locally is the smallest correct deployment: nothing new to learn, easy
to debug, cheap. The hardening notes above are the upgrade path to a managed,
production setup (ECS Fargate + RDS + ALB) when traffic justifies the extra moving parts.
