"""Write the seed eval questions (hand-drafted against the docs) to data/eval/questions.jsonl.

Kept as a script so the seed set is easy to read and edit in one place. Every quote must be
copied verbatim from the cleaned corpus; `rageval validate` checks that.
"""

import json
from pathlib import Path

# The first 30 questions were approved by the author; later ones are checked by Claude only.
APPROVED_BY_AUTHOR = 30
NOTE = "Hand-drafted against the docs; every quote checked verbatim with `rageval validate`."

Q = [
    ("factoid", "What is the maximum size of data a ConfigMap can hold?",
     [("concepts/configuration/configmap", "The data stored in a\nConfigMap cannot exceed 1 MiB.")],
     "1 MiB."),
    ("factoid", "How many old ReplicaSets does a Deployment keep for rollback by default?",
     [("concepts/workloads/controllers/deployment",
       "You can set `.spec.revisionHistoryLimit` field in a Deployment to specify how many old "
       "ReplicaSets for\nthis Deployment you want to retain. The rest will be garbage-collected "
       "in the background. By default,\nit is 10.")],
     "10, controlled by .spec.revisionHistoryLimit."),
    ("conceptual", "Are Kubernetes Secrets encrypted at rest by default?",
     [("concepts/configuration/secret",
       "Kubernetes Secrets are, by default, stored unencrypted in the API server's underlying "
       "data store\n(etcd).",
       [("tasks/administer-cluster/encrypt-data",
         "If you are running the kube-apiserver without the `--encryption-provider-config` "
         "command line\nargument, you do not have encryption at rest enabled.")])],
     "No. By default they are stored unencrypted in etcd."),
    ("procedural",
     "How do I put plain text values in a Secret manifest without base64-encoding them myself?",
     [("concepts/configuration/secret",
       "If the conversion to base64 string is not desirable, you can choose to specify\nthe "
       "`stringData` field instead, which accepts arbitrary strings as values.")],
     "Use the stringData field instead of data."),
    ("factoid", "Which namespaces exist in a brand new Kubernetes cluster?",
     [("concepts/overview/working-with-objects/namespaces",
       "Kubernetes starts with four initial namespaces:")],
     "Four: default, kube-node-lease, kube-public and kube-system."),
    ("factoid", "What port range are NodePort services allocated from by default?",
     [("concepts/services-networking/service",
       "allocates a port from a range specified by `--service-node-port-range` flag "
       "(default: 30000-32767).")],
     "30000-32767, set by --service-node-port-range."),
    ("factoid", "What is the default maxUnavailable for a Deployment rolling update?",
     [("concepts/workloads/controllers/deployment",
       "By default,\n  it ensures that at least 75% of the desired number of Pods are up "
       "(25% max unavailable).",
       [("concepts/workloads/controllers/deployment",
         "This depends on the rollingUpdate parameters (`maxUnavailable` specifically) that you "
         "have specified. Kubernetes by default sets the value to 25%."),
        ("reference/kubernetes-api/apps/deployment-v1",
         "This can not be 0 if MaxSurge is 0. Defaults to 25%.")])],
     "25%."),
    ("conceptual", "If no NetworkPolicy selects a pod, which inbound connections are allowed?",
     [("concepts/services-networking/network-policies",
       "By default, a pod is non-isolated for ingress; all inbound connections are allowed.")],
     "All of them; the pod is non-isolated for ingress."),
    ("comparison", "What is the difference between a Role and a ClusterRole in RBAC?",
     [("reference/access-authn-authz/rbac",
       "A Role always sets permissions within a particular namespace;\nwhen you create a Role, "
       "you have to specify the namespace it belongs in.\n\nClusterRole, by contrast, is a "
       "non-namespaced resource.")],
     "A Role is namespaced; a ClusterRole is cluster-scoped (non-namespaced)."),
    ("conceptual",
     "What happens to running pods that don't tolerate a NoExecute taint added to their node?",
     [("concepts/scheduling-eviction/taint-and-toleration",
       "Pods that do not tolerate the taint are evicted immediately")],
     "They are evicted immediately."),
    ("procedural", "How do I take a snapshot backup of etcd?",
     [("tasks/administer-cluster/configure-upgrade-etcd",
       "ETCDCTL_API=3 etcdctl --endpoints $ENDPOINT snapshot save snapshot.db")],
     "Run etcdctl snapshot save, e.g. ETCDCTL_API=3 etcdctl --endpoints $ENDPOINT snapshot "
     "save snapshot.db."),
    ("procedural", "How can I safely evict all pods from a node before doing kernel maintenance?",
     [("tasks/administer-cluster/safely-drain-node",
       "You can use `kubectl drain` to safely evict all of your pods from a\nnode before you "
       "perform maintenance on the node (e.g. kernel upgrade,\nhardware maintenance, etc.).")],
     "Use kubectl drain on the node."),
    ("conceptual", "How are init containers different from regular containers?",
     [("concepts/workloads/pods/init-containers",
       "* Init containers always run to completion.\n* Each init container must complete "
       "successfully before the next one starts.")],
     "They always run to completion, and each must succeed before the next starts."),
    ("conceptual", "When should I use a StatefulSet instead of a Deployment?",
     [("concepts/workloads/controllers/statefulset",
       "* Stable, unique network identifiers.\n* Stable, persistent storage.\n* Ordered, "
       "graceful deployment and scaling.\n* Ordered, automated rolling updates.")],
     "When the app needs stable network identities, stable persistent storage, or ordered "
     "deployment, scaling and rolling updates."),
    ("factoid", "What does concurrencyPolicy Forbid do on a CronJob?",
     [("concepts/workloads/controllers/cron-jobs",
       "* `Forbid`: The CronJob does not allow concurrent runs; if it is time for a new Job run "
       "and the\n  previous Job run hasn't finished yet, the CronJob skips the new Job run.")],
     "It prevents concurrent runs: if the previous Job is still running, the new run is skipped."),
    ("factoid", "What is the default concurrency policy for a CronJob?",
     [("concepts/workloads/controllers/cron-jobs",
       "* `Allow` (default): The CronJob allows concurrently running Jobs")],
     "Allow."),
    ("procedural",
     "How do I attach a debugging shell to a running pod that has no shell in its image?",
     [("tasks/debug/debug-application/debug-running-pod",
       "kubectl debug -it ephemeral-demo --image=busybox:1.28 --target=ephemeral-demo")],
     "Use kubectl debug -it <pod> --image=busybox --target=<container> to add an ephemeral "
     "container."),
    ("conceptual", "What does a DaemonSet guarantee?",
     [("concepts/workloads/controllers/daemonset",
       "A _DaemonSet_ ensures that all (or some) Nodes run a copy of a Pod.")],
     "That all (or some) nodes run a copy of a Pod."),
    ("factoid", "Can I upgrade a kubeadm cluster directly from 1.29 to 1.31?",
     [("tasks/administer-cluster/kubeadm/kubeadm-upgrade",
       "Skipping MINOR versions\nwhen upgrading is unsupported.")],
     "No, skipping minor versions is unsupported; upgrade one minor version at a time."),
    ("factoid", "How long does Kubernetes wait for a pod to terminate gracefully by default?",
     [("concepts/workloads/pods/pod-lifecycle",
       "A Pod is granted a term to terminate gracefully, which defaults to 30 seconds.")],
     "30 seconds."),
    ("conceptual", "Is data in an emptyDir volume lost when a container crashes?",
     [("concepts/storage/volumes",
       "A container crashing does *not* remove a Pod from a node. The data in an `emptyDir` "
       "volume\nis safe across container crashes.")],
     "No. emptyDir data survives container crashes; it is deleted when the Pod leaves the node."),
    ("conceptual", "Can a container use more memory than its request?",
     [("concepts/configuration/manage-resources-containers",
       "If the node where a Pod is running has enough of a resource available, it's possible "
       "(and\nallowed) for a container to use more resource than its `request` for that "
       "resource specifies.")],
     "Yes, if the node has spare capacity; the limit, not the request, is the cap."),
    ("procedural", "Why can't I delete a static Pod with kubectl, and how do I restart one?",
     [("tasks/administer-cluster/kubeadm/kubeadm-certs",
       "Static Pods are managed by the local kubelet\nand not by the API Server, thus kubectl "
       "cannot be used to delete and restart them.\nTo restart a static Pod you can temporarily "
       "remove its manifest file from `/etc/kubernetes/manifests/`")],
     "They are managed by the local kubelet, not the API server; restart one by temporarily "
     "removing its manifest from the static pod directory."),
    ("factoid", "How often does the Horizontal Pod Autoscaler check metrics by default?",
     [("concepts/workloads/autoscaling/horizontal-pod-autoscale",
       "The interval is set by the\n`--horizontal-pod-autoscaler-sync-period` parameter to the"
       "\n`kube-controller-manager`\n(and the default interval is 15 seconds).")],
     "Every 15 seconds by default."),
    ("factoid", "Which PersistentVolume access mode lets only a single pod mount the volume?",
     [("concepts/storage/persistent-volumes", "For single pod access, please see ReadWriteOncePod.")],
     "ReadWriteOncePod."),
    ("multi_hop",
     "If a NoExecute taint evicts my pod, what happens to the data in its emptyDir volume?",
     [("concepts/scheduling-eviction/taint-and-toleration",
       "Pods that do not tolerate the taint are evicted immediately"),
      ("concepts/storage/volumes",
       "When a Pod is removed from a node for\nany reason, the data in the `emptyDir` is "
       "deleted permanently.")],
     "The pod is evicted immediately, and because it leaves the node its emptyDir data is "
     "deleted permanently."),
    ("unanswerable",
     "What is the default CPU limit Kubernetes assigns to every container in a new cluster?", [],
     "Not answerable: there is no cluster-wide default CPU limit unless a LimitRange sets one."),
    ("unanswerable", "How much does a managed EKS control plane cost per hour?", [],
     "Not in the Kubernetes docs."),
    ("unanswerable",
     "What is the default number of Deployment replicas when the cluster has over 100 nodes?",
     [], "Not answerable as asked: the replica default does not depend on cluster size."),
    ("unanswerable", "Which Helm chart version ships with Kubernetes 1.30?", [],
     "Not answerable: Helm is not part of Kubernetes releases."),
    # Second batch: more comparison, multi-hop and unanswerable questions, which LLM drafting
    # from a single passage cannot produce.
    ("comparison", "What's the difference between a liveness probe and a readiness probe?",
     [("concepts/workloads/pods/pod-lifecycle",
       "Liveness probes determine when to restart a container."),
      ("concepts/workloads/pods/pod-lifecycle",
       "Readiness probes determine when a container is ready to accept traffic.")],
     "A failing liveness probe gets the container restarted; a failing readiness probe only "
     "stops it receiving traffic."),
    ("comparison", "How does a NodePort Service differ from a LoadBalancer Service?",
     [("concepts/services-networking/service",
       "Exposes the Service on each Node's IP at a static port (the `NodePort`)."),
      ("concepts/services-networking/service",
       "Exposes the Service externally using an external load balancer.")],
     "NodePort opens a static port on every node's IP; LoadBalancer exposes the Service through "
     "an external load balancer that you or a cloud provider must supply."),
    ("comparison",
     "What is the difference between required and preferred node affinity?",
     [("concepts/scheduling-eviction/assign-pod-node",
       "- `requiredDuringSchedulingIgnoredDuringExecution`: The scheduler can't\n  schedule the "
       "Pod unless the rule is met."),
      ("concepts/scheduling-eviction/assign-pod-node",
       "- `preferredDuringSchedulingIgnoredDuringExecution`: The scheduler tries to\n  find a "
       "node that meets the rule. If a matching node is not available, the\n  scheduler still "
       "schedules the Pod.")],
     "Required affinity is a hard rule the Pod cannot be scheduled without; preferred affinity "
     "is a soft preference and the Pod is still scheduled if no node matches."),
    ("comparison", "Retain vs Delete reclaim policy: what happens to the underlying storage?",
     [("concepts/storage/persistent-volumes",
       "The `Retain` reclaim policy allows for manual reclamation of the resource."),
      ("concepts/storage/persistent-volumes",
       "For volume plugins that support the `Delete` reclaim policy, deletion removes\nboth the "
       "PersistentVolume object from Kubernetes, as well as the associated\nstorage asset in "
       "the external infrastructure.")],
     "Retain keeps the volume for manual reclamation; Delete removes both the PersistentVolume "
     "and the storage asset."),
    ("comparison", "How is a PreferNoSchedule taint different from NoSchedule?",
     [("concepts/scheduling-eviction/taint-and-toleration",
       "`PreferNoSchedule` is a \"preference\" or \"soft\" version of `NoSchedule`.")],
     "PreferNoSchedule is a soft version: the scheduler tries to avoid the node but it is not "
     "guaranteed."),
    ("comparison", "When should I use a Secret rather than a ConfigMap?",
     [("concepts/configuration/secret",
       "Secrets are similar to ConfigMaps\nbut are specifically intended to hold confidential data.",
       [("concepts/configuration",
         "Secrets are similar to ConfigMaps\nbut are specifically intended to hold confidential "
         "data.")])],
     "Use a Secret for confidential data such as passwords, tokens or keys."),
    ("comparison", "Should a new cluster use Ingress or the Gateway API?",
     [("concepts/services-networking/ingress-controllers",
       "The Kubernetes project recommends using Gateway instead of\nIngress.",
       [("concepts/services-networking/gateway",
         "Gateway API is the successor to the Ingress API.")])],
     "Gateway API: the project recommends it over Ingress, whose API is frozen."),
    ("comparison", "What is the difference between a ClusterIP and an ExternalName Service?",
     [("concepts/services-networking/service",
       "Exposes the Service on a cluster-internal IP. Choosing this value\n  makes the Service "
       "only reachable from within the cluster."),
      ("concepts/services-networking/service",
       "Maps the Service to the contents of the `externalName` field")],
     "ClusterIP gives an internal IP reachable only in the cluster; ExternalName just returns a "
     "DNS CNAME to an external hostname, with no proxying."),
    ("conceptual", "I created an Ingress but nothing routes traffic. What am I missing?",
     [("concepts/services-networking/ingress",
       "You must have an Ingress controller\nto satisfy an Ingress. Only creating an Ingress "
       "resource has no effect.")],
     "An Ingress controller; the Ingress resource alone has no effect."),
    ("multi_hop", "My Job's Pod template doesn't set restartPolicy. Is that valid for a Job?",
     [("concepts/workloads/controllers/job",
       "`Job` is *only* appropriate\nfor pods with `RestartPolicy` equal to `OnFailure` or "
       "`Never`."),
      ("concepts/workloads/controllers/job",
       "If `RestartPolicy` is not set, the default value is `Always`.")],
     "No: unset means Always, but a Job only works with OnFailure or Never, so set one "
     "explicitly."),
    ("multi_hop",
     "With both a startup probe and a liveness probe, when does the liveness probe start and "
     "what does a later failure do?",
     [("concepts/workloads/pods/pod-lifecycle",
       "If a startup probe is configured, Kubernetes does not execute liveness or\nreadiness "
       "probes until the startup probe succeeds"),
      ("concepts/workloads/pods/pod-lifecycle",
       "Liveness probes determine when to restart a container.")],
     "Only after the startup probe succeeds; after that a liveness failure restarts the "
     "container."),
    ("conceptual",
     "After deleting a PVC whose volume has the Retain policy, can a new claim bind that volume?",
     [("concepts/storage/persistent-volumes",
       "When the PersistentVolumeClaim is deleted, the PersistentVolume still exists\nand the "
       "volume is considered \"released\". But it is not yet available for")],
     "Not immediately: the volume is released but must be manually reclaimed first."),
    ("conceptual",
     "What requests does a Pod without resource settings get in a namespace with a LimitRange?",
     [("concepts/policy/limit-range",
       "First, the LimitRange admission controller applies default request and limit values\n  "
       "for all Pods (and their containers) that do not set compute resource requirements.")],
     "The LimitRange's default requests and limits are applied to it."),
    ("unanswerable", "What uptime SLA does the Kubernetes project guarantee for the control plane?",
     [], "Not answerable: upstream Kubernetes offers no SLA."),
    ("unanswerable", "How do I configure service autoscaling in Docker Swarm?", [],
     "Out of scope: Docker Swarm is not covered by the Kubernetes docs."),
    ("unanswerable", "What is the price of a GKE Autopilot pod per vCPU hour?", [],
     "Not in the Kubernetes docs."),
    ("unanswerable",
     "How many nodes did Google's largest internal Borg cell run in 2019?", [],
     "Not in the Kubernetes docs."),
    ("unanswerable", "Which AWS EC2 instance type does EKS recommend for etcd?", [],
     "Not in the Kubernetes docs."),
    ("unanswerable", "What is the maximum number of Helm releases allowed per namespace?", [],
     "Not in the Kubernetes docs: Helm is a separate project."),
    ("unanswerable", "How do I enable GPU overclocking through the kubelet?", [],
     "Not answerable: the kubelet has no such feature."),
    ("unanswerable", "Which Kubernetes version introduced the kubectl `deploy-wizard` command?",
     [], "Not answerable: kubectl has no deploy-wizard command."),
    ("unanswerable", "What default username and password does the Kubernetes API server ship with?",
     [], "Not answerable: the API server does not ship with default credentials."),
    ("unanswerable", "How do I migrate a Nomad job file to a Kubernetes Deployment?", [],
     "Not in the Kubernetes docs."),
    ("unanswerable", "What is the Kubernetes certification exam passing score for CKA in 2026?",
     [], "Not in the Kubernetes docs."),
]


def _evidence(doc_id, quote, alternatives=()):
    """Alternatives are the same fact stated on another page; retrieving any one counts."""
    e = {"doc_id": doc_id, "quote": quote}
    if alternatives:
        e["alternatives"] = [{"doc_id": d, "quote": q} for d, q in alternatives]
    return e


if __name__ == "__main__":
    out = Path("data/eval/questions.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, (typ, question, evidence, ref) in enumerate(Q, start=1):
            f.write(json.dumps({
                "qid": f"h{i:04d}", "question": question, "type": typ,
                "answerable": typ != "unanswerable",
                "evidence": [_evidence(*e) for e in evidence],
                "reference_answer": ref, "origin": "hand", "reviewed": True,
                "reviewer": "Claude, approved by Abdul" if i <= APPROVED_BY_AUTHOR else "claude",
                "notes": NOTE,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(Q)} questions to {out}")
