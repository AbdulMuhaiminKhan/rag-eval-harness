# Results: ablation_full

171 answerable questions (186 total). Means with 95% bootstrap confidence intervals over questions.

| config | recall@5 | mrr@10 | ndcg@10 | doc_recall@5 | p50 latency | chunks |
|---|---|---|---|---|---|---|
| fixed__hybrid__ce__minilm | 0.787 [0.73, 0.85] | 0.657 [0.60, 0.72] | 0.706 [0.65, 0.76] | 0.944 | 219 ms | 15945 |
| fixed__hybrid__ce__bge-small | 0.784 [0.73, 0.84] | 0.653 [0.60, 0.71] | 0.702 [0.65, 0.75] | 0.936 | 222 ms | 15945 |
| fixed__bm25__ce__bge-small | 0.781 [0.72, 0.84] | 0.655 [0.60, 0.71] | 0.703 [0.65, 0.76] | 0.939 | 176 ms | 15945 |
| fixed__bm25__ce__minilm | 0.781 [0.72, 0.84] | 0.655 [0.60, 0.71] | 0.703 [0.65, 0.76] | 0.939 | 158 ms | 15945 |
| semantic__hybrid__norerank__bge-small | 0.775 [0.71, 0.84] | 0.620 [0.56, 0.68] | 0.671 [0.61, 0.73] | 0.927 | 72 ms | 13400 |
| semantic__hybrid__ce__minilm | 0.772 [0.71, 0.83] | 0.659 [0.60, 0.72] | 0.707 [0.65, 0.76] | 0.933 | 243 ms | 13400 |
| fixed__dense__ce__minilm | 0.769 [0.71, 0.83] | 0.630 [0.57, 0.69] | 0.669 [0.61, 0.73] | 0.915 | 159 ms | 15945 |
| semantic__hybrid__ce__bge-small | 0.769 [0.70, 0.83] | 0.659 [0.60, 0.72] | 0.705 [0.65, 0.76] | 0.927 | 276 ms | 13400 |
| fixed__dense__ce__bge-small | 0.763 [0.70, 0.82] | 0.616 [0.56, 0.68] | 0.659 [0.60, 0.71] | 0.906 | 176 ms | 15945 |
| semantic__bm25__ce__bge-small | 0.760 [0.70, 0.82] | 0.653 [0.59, 0.71] | 0.696 [0.64, 0.75] | 0.927 | 240 ms | 13400 |
| semantic__bm25__ce__minilm | 0.760 [0.70, 0.82] | 0.653 [0.59, 0.71] | 0.696 [0.64, 0.75] | 0.927 | 242 ms | 13400 |
| semantic__dense__ce__minilm | 0.743 [0.68, 0.80] | 0.624 [0.56, 0.68] | 0.666 [0.61, 0.72] | 0.886 | 205 ms | 13400 |
| semantic__bm25__norerank__bge-small | 0.737 [0.67, 0.80] | 0.594 [0.53, 0.66] | 0.643 [0.58, 0.70] | 0.918 | 35 ms | 13400 |
| semantic__bm25__norerank__minilm | 0.737 [0.67, 0.80] | 0.594 [0.53, 0.66] | 0.643 [0.58, 0.70] | 0.918 | 40 ms | 13400 |
| semantic__dense__ce__bge-small | 0.737 [0.67, 0.80] | 0.613 [0.55, 0.68] | 0.659 [0.60, 0.72] | 0.868 | 220 ms | 13400 |
| fixed__bm25__norerank__bge-small | 0.716 [0.65, 0.78] | 0.546 [0.49, 0.61] | 0.609 [0.55, 0.67] | 0.918 | 51 ms | 15945 |
| fixed__bm25__norerank__minilm | 0.716 [0.65, 0.78] | 0.546 [0.49, 0.61] | 0.609 [0.55, 0.67] | 0.918 | 53 ms | 15945 |
| fixed__hybrid__norerank__bge-small | 0.716 [0.65, 0.78] | 0.568 [0.51, 0.63] | 0.630 [0.58, 0.68] | 0.921 | 80 ms | 15945 |
| semantic__hybrid__norerank__minilm | 0.716 [0.65, 0.78] | 0.569 [0.51, 0.63] | 0.628 [0.57, 0.68] | 0.930 | 62 ms | 13400 |
| fixed__hybrid__norerank__minilm | 0.705 [0.64, 0.77] | 0.578 [0.52, 0.64] | 0.636 [0.58, 0.69] | 0.912 | 64 ms | 15945 |
| semantic__dense__norerank__bge-small | 0.696 [0.63, 0.76] | 0.519 [0.46, 0.58] | 0.572 [0.51, 0.63] | 0.836 | 23 ms | 13400 |
| fixed__dense__norerank__bge-small | 0.635 [0.56, 0.70] | 0.501 [0.44, 0.57] | 0.557 [0.50, 0.62] | 0.839 | 23 ms | 15945 |
| semantic__dense__norerank__minilm | 0.629 [0.56, 0.70] | 0.442 [0.38, 0.50] | 0.497 [0.44, 0.55] | 0.801 | 15 ms | 13400 |
| fixed__dense__norerank__minilm | 0.582 [0.51, 0.65] | 0.430 [0.37, 0.49] | 0.486 [0.43, 0.54] | 0.807 | 16 ms | 15945 |

**fixed__hybrid__ce__minilm** vs **fixed__dense__norerank__minilm** on recall@5: +0.205 (paired 95% CI [+0.135, +0.272]).

## By question type (fixed__hybrid__ce__minilm)

| type | n | recall@5 | mrr@10 |
|---|---|---|---|
| comparison | 17 | 0.706 [0.47, 0.88] | 0.521 [0.35, 0.70] |
| conceptual | 48 | 0.750 [0.62, 0.85] | 0.605 [0.48, 0.72] |
| factoid | 37 | 0.838 [0.70, 0.95] | 0.741 [0.61, 0.85] |
| multi_hop | 9 | 0.611 [0.33, 0.83] | 0.651 [0.39, 0.91] |
| procedural | 60 | 0.833 [0.73, 0.93] | 0.685 [0.59, 0.78] |

## By question origin (recall@5)

| config | hand-written | LLM-drafted |
|---|---|---|
| fixed__hybrid__ce__minilm | 0.744 [0.62, 0.87] (n=45) | 0.802 [0.73, 0.87] (n=126) |
| fixed__dense__norerank__minilm | 0.456 [0.31, 0.59] (n=45) | 0.627 [0.54, 0.71] (n=126) |

## Failure analysis (fixed__hybrid__ce__minilm)

Missed in top 10: 19 (right page, wrong passage: 14, wrong page: 5)

| qid | type | failure | question | gold page | top retrieved page |
|---|---|---|---|---|---|
| h0032 | comparison | right page, wrong passage | How does a NodePort Service differ from a LoadBalancer Service? | `concepts/services-networking/service, concepts/services-networking/service` | `concepts/services-networking/service` |
| g0009 | conceptual | right page, wrong passage | When should I use manual scaling instead of automatic scaling for a Deployment? | `tasks/run-application/scale-deployment` | `tasks/run-application/scale-deployment` |
| g0026 | conceptual | wrong page | How can I use a variable in an expression within Kubernetes admission webhook validation? | `reference/kubernetes-api/definitions/variable-v1-admissionregistration` | `concepts/cluster-administration/admission-webhooks-good-practices` |
| g0029 | conceptual | right page, wrong passage | What is the recommended method to secure communication between the Kubernetes control plane and nodes? | `concepts/architecture/control-plane-node-communication` | `reference/config-api/kubeadm-config.v1beta4` |
| g0031 | procedural | right page, wrong passage | How do I suggest a new content idea to the Kubernetes documentation team? | `contribute/suggesting-improvements` | `contribute/suggesting-improvements` |
| g0040 | procedural | right page, wrong passage | How can I ensure that a pod runs on a regular cluster node and won't be evicted even if the node is under high utilization? | `tasks/administer-cluster/guaranteed-scheduling-critical-addon-pods` | `tasks/administer-cluster/guaranteed-scheduling-critical-addon-pods` |
| g0043 | conceptual | right page, wrong passage | What types of probes can be used to check the status of a container in Kubernetes? | `concepts/workloads/pods/probes` | `concepts/workloads/pods/probes` |
| g0051 | conceptual | right page, wrong passage | Why can't I update the role in a ClusterRoleBinding after it has been created? | `reference/access-authn-authz/rbac` | `reference/access-authn-authz/rbac` |
| g0055 | procedural | right page, wrong passage | How can I check if a Job has completed successfully using kubectl? | `tasks/job/fine-parallel-processing-work-queue` | `concepts/workloads/controllers/job` |
| g0056 | factoid | right page, wrong passage | Can I use different patch types when customizing the control plane components? | `setup/production-environment/tools/kubeadm/control-plane-flags` | `setup/production-environment/tools/kubeadm/control-plane-flags` |
| g0062 | conceptual | right page, wrong passage | How can I ensure that the client's source IP is preserved when accessing a Kubernetes Service from outside the cluster? | `tutorials/services/source-ip` | `concepts/services-networking/service` |
| g0071 | procedural | wrong page | How can I prevent new pods from being scheduled on a specific node in Kubernetes? | `reference/kubectl/generated/kubectl_cordon` | `concepts/workloads/pods/pod-lifecycle` |
| g0089 | procedural | right page, wrong passage | How do I clean up the persistent volume claims and volumes after using StatefulSets in this tutorial? | `tutorials/stateful-application/basic-stateful-set` | `tutorials/stateful-application/cassandra` |
| g0096 | conceptual | wrong page | Do I need to use WSL on Windows to develop Kubernetes documentation? | `contribute/generate-ref-docs/prerequisites-ref-docs` | `tutorials/kubernetes-basics/deploy-app/deploy-intro` |
| g0105 | procedural | wrong page | What conditions must be met for the CSIMigrationAzureFileComplete feature gate to be enabled? | `reference/command-line-tools-reference/feature-gates/CSIMigrationAzureFileComplete` | `reference/kubernetes-api/core/persistent-volume-claim-v1` |
| g0107 | factoid | right page, wrong passage | What happens if I set the externalTrafficPolicy to Local on a service running on GKE/GCE? | `tutorials/services/source-ip` | `tutorials/services/source-ip` |
| g0117 | conceptual | wrong page | What is the purpose of Cluster API? | `setup/production-environment/tools` | `reference/glossary/cluster-operator` |
| g0129 | conceptual | right page, wrong passage | What should I consider for sensitive workloads in Kubernetes? | `concepts/security/application-security-checklist` | `concepts/security/application-security-checklist` |
| g0135 | procedural | right page, wrong passage | How can I see the details of a specific revision using kubectl rollout history? | `reference/kubectl/generated/kubectl_rollout/kubectl_rollout_history` | `tasks/manage-daemon/rollback-daemon-set` |

![quality vs latency](quality_vs_latency.png)
