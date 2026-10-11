"""Write the seed eval questions (hand-drafted against the docs) to data/eval/questions.jsonl.

Kept as a script so the seed set is easy to read and edit in one place. Every quote must be
copied verbatim from the cleaned corpus; `rageval validate` checks that.
"""

import json
from pathlib import Path

REVIEWER = "Claude, set approved by Abdul"
NOTE = "Seed question drafted against the docs; evidence re-checked and set approved by the author."

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
                "reference_answer": ref, "origin": "hand", "reviewed": True, "reviewer": REVIEWER,
                "notes": NOTE,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(Q)} questions to {out}")
