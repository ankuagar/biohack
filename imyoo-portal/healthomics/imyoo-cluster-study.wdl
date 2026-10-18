version 1.0

workflow ImYooClusterStudy {
  input {
    File h5ad
    String container_image
    String resolutions = "0.2,0.4,0.6,0.8,1.0"
    String neighbors_list = "15"
    Int max_cells = 100000
  }

  call ClusterStudy {
    input:
      h5ad = h5ad,
      container_image = container_image,
      resolutions = resolutions,
      neighbors_list = neighbors_list,
      max_cells = max_cells
  }

  output {
    File study_cells = ClusterStudy.study_cells
    File study_summary = ClusterStudy.study_summary
    File study_stability = ClusterStudy.study_stability
    File study_info = ClusterStudy.study_info
  }
}

task ClusterStudy {
  input {
    File h5ad
    String container_image
    String resolutions
    String neighbors_list
    Int max_cells
  }

  command <<<
    set -euo pipefail
    mkdir -p outputs
    python /opt/imyoo/cluster_study.py \
      --input ~{h5ad} \
      --out-dir outputs \
      --resolutions '~{resolutions}' \
      --neighbors-list '~{neighbors_list}' \
      --max-cells ~{max_cells}
  >>>

  output {
    File study_cells = "outputs/cluster_study_cells.parquet"
    File study_summary = "outputs/cluster_study_summary.parquet"
    File study_stability = "outputs/cluster_study_stability.parquet"
    File study_info = "outputs/cluster_study_info.json"
  }

  runtime {
    container: container_image
    cpu: 8
    memory: "64 GB"
    preemptible: 0
  }
}
