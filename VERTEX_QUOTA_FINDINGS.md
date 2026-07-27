# Vertex AI training quota, measured

Evidence level: **live platform**. Every limit below came from a submission
that either succeeded or returned HTTP 429 naming the exact metric.

## What the quota listing implies versus what submission proves

A quota listing showed `restricted_image_training_nvidia_a100_80gb_gpus`
at limit 8, which suggested A100 80 GB was available. Submitting a custom
container proved otherwise: that quota applies to restricted images, not to
custom training jobs.

| Metric named in a 429 | Effective limit |
|---|---:|
| `custom_model_training_nvidia_a100_gpus` | 0 |
| `custom_model_training_nvidia_a100_80gb_gpus` | 0 |
| `custom_model_training_preemptible_nvidia_a100_80gb_gpus` | 0 |
| `custom_model_training_nvidia_l4_gpus` | 0 |
| `custom_model_training_preemptible_nvidia_a100_gpus` | 8 |

Only **preemptible A100 40 GB** is usable for custom containers, so jobs must
request `SPOT` scheduling on `a2-highgpu-1g`.

## Consequences for the pipeline

Checkpoints must leave the boot disk continuously, because preemptible
capacity is reclaimed without warning. `gcs_mirror.sh` syncs every 60
seconds and `EVAL_STEPS` is 100, bounding loss to roughly 29 minutes of
progress at the measured 17.26 s/it.

A `PENDING` job with no error is waiting for capacity rather than failing;
preemptible A100 in us-central1 queued for over ten minutes during this
session.

## Permissions added during setup

The service account began with `roles/aiplatform.admin` but could not read
logs or run builds. Each was granted only after a specific denial:

| Role | Unblocked |
|---|---|
| `roles/logging.privateLogViewer` | reading container stdout, which is the only way failures were diagnosable |
| `roles/cloudbuild.builds.editor` | submitting the trainer image build |
| `roles/iam.serviceAccountUser` | acting as the Cloud Build service account |

Cloud Logging access mattered most: mirrored logs truncated to 28 to 55
bytes on early failures, and three runs were misdiagnosed before real
stdout became readable.

## Quota increase requested

A preference for `CustomModelTrainingPreemptibleA100GPUsPerProjectPerRegion`
at 8 in us-central1 was submitted as `ct-train-a100-preempt`. Requests are
reviewed asynchronously, so this is pending rather than granted.