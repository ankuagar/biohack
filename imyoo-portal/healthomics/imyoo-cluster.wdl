version 1.0

workflow ImYooCluster {
  input {
    File h5ad
    String container_image
    Float resolution = 0.6
    Int neighbors = 15
    Int markers = 10
    Int max_cells = 100000
  }

  call Cluster {
    input:
      h5ad = h5ad,
      container_image = container_image,
      resolution = resolution,
      neighbors = neighbors,
      markers = markers,
      max_cells = max_cells
  }

  output {
    File cluster_cells = Cluster.cluster_cells
    File cluster_summary = Cluster.cluster_summary
    File cluster_markers = Cluster.cluster_markers
    File cluster_info = Cluster.cluster_info
  }
}

task Cluster {
  input {
    File h5ad
    String container_image
    Float resolution
    Int neighbors
    Int markers
    Int max_cells
  }

  command <<<
    set -euo pipefail
    mkdir -p outputs
    python /opt/imyoo/cluster_imyoo.py \
      --input ~{h5ad} \
      --out-dir outputs \
      --resolution ~{resolution} \
      --neighbors ~{neighbors} \
      --markers ~{markers} \
      --max-cells ~{max_cells}
  >>>

  output {
    File cluster_cells = "outputs/cluster_cells.parquet"
    File cluster_summary = "outputs/cluster_summary.parquet"
    File cluster_markers = "outputs/cluster_markers.parquet"
    File cluster_info = "outputs/cluster_info.json"
  }

  runtime {
    container: container_image
    cpu: 8
    memory: "64 GB"
    preemptible: 0
  }
}
