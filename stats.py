# pylint: disable=c-extension-no-member,fixme
'''
Defining the class that computes cluster statistics
'''

import json
import os
from typing import Dict, List

import networkit as nk
import pandas as pd
import typer
from hm01.graph import Graph, IntangibleSubgraph, RealizedSubgraph
from hm01.mincut import viecut
from numpy import log2, log10


class Statistics:
    '''
    Class to Compute Cluster Statistics
    '''

    def __init__(
        self,
        input_file,
        existing_clustering,
        resolution=0,
        output=None,
    ):

        self.input = input_file
        self.existing_clustering = existing_clustering
        self.resolution = resolution
        # self.universal_before = universal_before

        if output is None or output == "":
            base, _ = os.path.splitext(existing_clustering)
            self.outfile = base + '_stats.csv'
        else:
            self.outfile = output

        base, _ = os.path.splitext(self.existing_clustering)
        self.summary_outfile = base + 'summary_stats.csv'

        self.clusters = None
        self.realized_clusters = None

        self.cluster_stats = None
        self.overall_cluster_stats = None
        self.summary_stats = None

        self.global_graph = None
        self.ids = None

        self.n = None
        self.ns = None

        self.m = None
        self.ms = None

        self.modularity = None
        self.modularities = None

        self.cpms = None

        self.mincut_results = None
        self.mincuts = None
        self.mincuts_normalized = None
        self.mincuts_normalized_log2 = None
        self.mincuts_normalized_sqrt = None

        self.conductances = None

    def from_tsv(self) -> List[RealizedSubgraph]:
        '''
        Getting cluster from a tsv file
        '''

        self.clusters = from_existing_clustering(
            self.existing_clustering).values()

        self.ids = [cluster.index for cluster in self.clusters]
        self.ns = [cluster.n() for cluster in self.clusters]

        # (VR) Load full graph into Graph object
        edgelist_reader = nk.graphio.EdgeListReader("\t", 0)
        nk_graph = edgelist_reader.read(self.input)

        self.global_graph = Graph(nk_graph, "")

        self.realized_clusters = [
            cluster.realize(self.global_graph) for cluster in self.clusters
        ]

    def to_csv(self):
        '''
        saving the stats to a csv
        '''

        self.cluster_stats.to_csv(self.outfile, index=False)

        return self.cluster_stats

    def to_summary_csv(self):
        '''
        saving the summary stats to a csv
        '''

        self.summary_stats.to_csv(self.summary_outfile, index=False)

        return self.summary_stats

    def compute_stats(self):
        '''
        computing cluster statistics
        '''

        self.ms = [
            cluster.count_edges(self.global_graph) for cluster in self.clusters
        ]

        self.modularities = [
            self.global_graph.modularity_of(cluster)
            for cluster in self.clusters
        ]

        if self.resolution != -1:
            self.cpms = [
                self.global_graph.cpm(cluster, self.resolution)
                for cluster in self.clusters
            ]

        # Only realized clusters from this point onward

        self.mincut_results = [
            viecut(cluster) for cluster in self.realized_clusters
        ]
        self.mincuts = [result[-1] for result in self.mincut_results]
        self.mincuts_normalized = [
            mincut / log10(self.ns[i]) for i, mincut in enumerate(self.mincuts)
        ]
        self.mincuts_normalized_log2 = [
            mincut / log2(self.ns[i]) for i, mincut in enumerate(self.mincuts)
        ]
        self.mincuts_normalized_sqrt = [
            mincut / (self.ns[i]**0.5 / 5)
            for i, mincut in enumerate(self.mincuts)
        ]

        self.conductances = []
        for _, cluster in enumerate(self.realized_clusters):
            self.conductances.append(cluster.conductance(self.global_graph))

        if self.resolution != -1:
            self.cluster_stats = pd.DataFrame(
                list(
                    zip(self.ids, self.ns, self.ms, self.modularities,
                        self.cpms, self.mincuts, self.mincuts_normalized,
                        self.mincuts_normalized_log2,
                        self.mincuts_normalized_sqrt, self.conductances)),
                columns=[
                    'cluster',
                    'n',
                    'm',
                    'modularity',
                    'cpm_score',
                    'connectivity',
                    'connectivity_normalized_log10(n)',
                    'connectivity_normalized_log2(n)',
                    'connectivity_normalized_sqrt(n)/5',
                    'conductance',
                ])
        else:
            self.cluster_stats = pd.DataFrame(
                list(
                    zip(
                        self.ids,
                        self.ns,
                        self.ms,
                        self.modularities,
                        self.mincuts,
                        self.mincuts_normalized,
                        self.mincuts_normalized_log2,
                        self.mincuts_normalized_sqrt,
                        self.conductances,
                    )),
                columns=[
                    'cluster',
                    'n',
                    'm',
                    'modularity',
                    'connectivity',
                    'connectivity_normalized_log10(n)',
                    'connectivity_normalized_log2(n)',
                    'connectivity_normalized_sqrt(n)/5',
                    'conductance',
                ])

        # Computing Overall Stats

        if self.resolution != -1:
            print([
                sum(self.ns),
                sum(self.ms),
                sum(self.modularities),
                sum(self.cpms),
            ])
            self.overall_cluster_stats = pd.DataFrame(
                [[
                    sum(self.ns),
                    sum(self.ms),
                    sum(self.modularities),
                    sum(self.cpms),
                ]],
                columns=['n', 'm', 'modularity', 'cpm_score'])
        else:
            print([sum(self.ns), sum(self.ms), sum(self.modularities)])
            self.overall_cluster_stats = pd.DataFrame(
                [[
                    sum(self.ns),
                    sum(self.ms),
                    sum(self.modularities),
                ]],
                columns=['n', 'm', 'modularity'])

    def basic_stats(self, column):
        '''
        computing basic summary statistics for the compute_summary function 
        '''
        column_min = column.min()
        column_max = column.max()
        column_med = column.median()
        column_q1 = column.quantile(0.25)
        column_q3 = column.quantile(0.75)
        column_mean = column.mean()

        return (
            column_min,
            column_max,
            column_med,
            column_q1,
            column_q3,
            column_mean,
        )

    def compute_summary(self) -> pd.DataFrame:
        '''
        computing summary stats
        '''

        # TODO: Refazer código com base no summarize.py
        # Erro Anterior: Confundir Overall com Summary

        # number of nodes
        self.n = self.overall_cluster_stats['n']

        # number of edges
        self.m = self.overall_cluster_stats['m']

        self.modularities = self.cluster_stats.iloc[:-1]['modularity']
        self.conductances = self.cluster_stats.iloc[:-1]['conductance']
        self.mincuts = self.cluster_stats.iloc[:-1]['connectivity']
        self.mincuts_normalized = self.cluster_stats.iloc[:-1][
            'connectivity_normalized_log10(n)']

        node_dist = self.cluster_stats.iloc[:-1]['n']

        # total number of nodes
        total_n = node_dist.sum()

        # total number of edges
        total_m = self.cluster_stats.iloc[:-1]['m'].sum()

        # number of nodes in clusters with 2 or more nodes
        total_n2 = node_dist[node_dist > 1].sum()

        # number of nodes in clusters with 10 or more nodes
        total_n11 = node_dist[node_dist > 10].sum()

        # min number of nodes in cluster (smallest one)
        (
            min_cluster,
            max_cluster,
            med_cluster,
            q1_cluster,
            q3_cluster,
            mean_cluster,
        ) = self.basic_stats(node_dist)

        self.modularity = self.overall_cluster_stats['modularity']

        (
            modularity_min,
            modularity_max,
            modularity_med,
            modularity_q1,
            modularity_q3,
            modularity_mean,
        ) = self.basic_stats(self.modularities)

        if self.resolution != -1:
            cpm_score = self.overall_cluster_stats['cpm_score']
            self.cpms = self.cluster_stats.iloc[:-1]['cpm_score']

            (
                cpm_min,
                cpm_max,
                cpm_med,
                cpm_q1,
                cpm_q3,
                cpm_mean,
            ) = self.basic_stats(self.cpms)

        (
            conductance_min,
            conductance_max,
            conductance_med,
            conductance_q1,
            conductance_q3,
            conductance_mean,
        ) = self.basic_stats(self.conductances)

        (
            mincuts_min,
            mincuts_max,
            mincuts_med,
            mincuts_q1,
            mincuts_q3,
            mincuts_mean,
        ) = self.basic_stats(self.mincuts)

        (
            mincuts_normalized_min,
            mincuts_normalized_max,
            mincuts_normalized_med,
            mincuts_normalized_q1,
            mincuts_normalized_q3,
            mincuts_normalized_mean,
        ) = self.basic_stats(self.mincuts)

        coverage_2 = round(total_n2 / self.n, 3)
        coverage_11 = round(total_n11 / self.n, 3)

        if self.resolution != -1:
            self.summary_stats = pd.Series({
                'network':
                self.outfile,
                'num_clusters':
                self.cluster_stats.shape[0] - 1,
                'network_n':
                self.n,
                'network_m':
                self.m,
                'total_n':
                total_n,
                'total_m':
                total_m,
                'cluster_size_dist': [
                    min_cluster,
                    q1_cluster,
                    med_cluster,
                    q3_cluster,
                    max_cluster,
                ],
                'mean_cluster_size':
                mean_cluster,
                'total_modularity':
                self.modularity,
                'modularity_dist': [
                    modularity_min,
                    modularity_q1,
                    modularity_med,
                    modularity_q3,
                    modularity_max,
                ],
                'modularity_mean':
                modularity_mean,
                'total_cpm_score':
                cpm_score,
                'cpm_dist': [
                    cpm_min,
                    cpm_q1,
                    cpm_med,
                    cpm_q3,
                    cpm_max,
                ],
                'cpm_mean':
                cpm_mean,
                'conductance_dist': [
                    conductance_min,
                    conductance_q1,
                    conductance_med,
                    conductance_q3,
                    conductance_max,
                ],
                'conductance_mean':
                conductance_mean,
                'mincuts_dist': [
                    mincuts_min,
                    mincuts_q1,
                    mincuts_med,
                    mincuts_q3,
                    mincuts_max,
                ],
                'mincuts_mean':
                mincuts_mean,
                'mincuts_normalized_dist': [
                    mincuts_normalized_min,
                    mincuts_normalized_q1,
                    mincuts_normalized_med,
                    mincuts_normalized_q3,
                    mincuts_normalized_max,
                ],
                'mincuts_mean_normalized':
                mincuts_normalized_mean,
                'node_coverage':
                coverage_2,
                'node_coverage_gr10':
                coverage_11,
            })
        else:
            self.summary_stats = pd.Series({
                'network':
                self.outfile,
                'num_clusters':
                self.cluster_stats.shape[0] - 1,
                'network_n':
                self.n,
                'network_m':
                self.m,
                'total_n':
                total_n,
                'total_m':
                total_m,
                'cluster_size_dist': [
                    min_cluster,
                    q1_cluster,
                    med_cluster,
                    q3_cluster,
                    max_cluster,
                ],
                'mean_cluster_size':
                mean_cluster,
                'total_modularity':
                self.modularity,
                'modularity_dist': [
                    modularity_min,
                    modularity_q1,
                    modularity_med,
                    modularity_q3,
                    modularity_max,
                ],
                'modularity_mean':
                modularity_mean,
                'conductance_dist': [
                    conductance_min,
                    conductance_q1,
                    conductance_med,
                    conductance_q3,
                    conductance_max,
                ],
                'conductance_mean':
                conductance_mean,
                'mincuts_dist': [
                    mincuts_min,
                    mincuts_q1,
                    mincuts_med,
                    mincuts_q3,
                    mincuts_max,
                ],
                'mincuts_mean':
                mincuts_mean,
                'mincuts_normalized_dist': [
                    mincuts_normalized_min,
                    mincuts_normalized_q1,
                    mincuts_normalized_med,
                    mincuts_normalized_q3,
                    mincuts_normalized_max,
                ],
                'mincuts_mean_normalized':
                mincuts_normalized_mean,
                'node_coverage':
                coverage_2,
                'node_coverage_gr10':
                coverage_11
            })


def from_existing_clustering(filepath) -> List[IntangibleSubgraph]:
    ''' 
    I just modified the original method to return a dict 
    mapping from index to clustering .
    '''
    # node_id cluster_id format
    clusters: Dict[str, IntangibleSubgraph] = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            node_id, cluster_id = line.split()
            clusters.setdefault(cluster_id, IntangibleSubgraph(
                [], cluster_id)).subset.append(int(node_id))
    return {key: val for key, val in clusters.items() if val.n() > 1}


def main(
        input_file: str = typer.Option(
            ...,
            "--input",
            "-i",
        ),
        existing_clustering: str = typer.Option(
            ...,
            "--existing-clustering",
            "-e",
        ),
        resolution: float = typer.Option(
            -1,
            "--resolution",
            "-g",
        ),
        universal_before: str = typer.Option(
            "",
            "--universal-before",
            "-ub",
        ),
        output: str = typer.Option(
            "",
            "--output",
            "-o",
        ),
):
    '''
    Main function for use in terminal
    '''

    if output == "":
        base, _ = os.path.splitext(existing_clustering)
        outfile = base + '_stats.csv'
    else:
        outfile = output

    print("Loading clusters...")
    clusters = from_existing_clustering(existing_clustering).values()
    ids = [cluster.index for cluster in clusters]
    ns = [cluster.n() for cluster in clusters]
    print("Done")

    print("Loading graph...")

    # (VR) Load full graph into Graph object
    edgelist_reader = nk.graphio.EdgeListReader("\t", 0)
    nk_graph = edgelist_reader.read(input_file)

    global_graph = Graph(nk_graph, "")
    ms = [cluster.count_edges(global_graph) for cluster in clusters]
    print("Done")

    # --------------------------------------------------
    # Statistics Computing

    print("Computing modularity...")
    modularities = [
        global_graph.modularity_of(cluster) for cluster in clusters
    ]
    print("Done")

    if resolution != -1:
        print("Computing CPM score...")
        cpms = [global_graph.cpm(cluster, resolution) for cluster in clusters]
        print("Done")

    print("Realizing clusters...")
    clusters = [cluster.realize(global_graph) for cluster in clusters]
    print("Done")

    print("Computing mincut...")
    mincut_results = [viecut(cluster) for cluster in clusters]
    mincuts = [result[-1] for result in mincut_results]
    mincuts_normalized = [
        mincut / log10(ns[i]) for i, mincut in enumerate(mincuts)
    ]
    mincuts_normalized_log2 = [
        mincut / log2(ns[i]) for i, mincut in enumerate(mincuts)
    ]
    mincuts_normalized_sqrt = [
        mincut / (ns[i]**0.5 / 5) for i, mincut in enumerate(mincuts)
    ]

    print("Done")

    print("Computing conductance...")
    conductances = []
    for i, cluster in enumerate(clusters):
        conductances.append(cluster.conductance(global_graph))
    print("Done")

    print("Computing overall stats...")
    m = global_graph.m()
    ids.append("Overall")
    modularities.append(sum(modularities))

    if resolution != -1:
        cpms.append(sum(cpms))

    ns.append(global_graph.n())
    ms.append(m)
    mincuts.append(None)
    mincuts_normalized.append(None)
    mincuts_normalized_log2.append(None)
    mincuts_normalized_sqrt.append(None)
    conductances.append(None)
    # ktruss_nodes.append(None)
    print("Done")

    # ----------------------------------------------------------
    # Output File

    print("Writing to output file...")

    if resolution != -1:
        df = pd.DataFrame(
            list(
                zip(
                    ids,
                    ns,
                    ms,
                    modularities,
                    cpms,
                    mincuts,
                    mincuts_normalized,
                    mincuts_normalized_log2,
                    mincuts_normalized_sqrt,
                    conductances,
                )),
            columns=[
                'cluster',
                'n',
                'm',
                'modularity',
                'cpm_score',
                'connectivity',
                'connectivity_normalized_log10(n)',
                'connectivity_normalized_log2(n)',
                'connectivity_normalized_sqrt(n)/5',
                'conductance',
            ],
        )
    else:
        df = pd.DataFrame(
            list(
                zip(
                    ids,
                    ns,
                    ms,
                    modularities,
                    mincuts,
                    mincuts_normalized,
                    mincuts_normalized_log2,
                    mincuts_normalized_sqrt,
                    conductances,
                )),
            columns=[
                'cluster',
                'n',
                'm',
                'modularity',
                'connectivity',
                'connectivity_normalized_log10(n)',
                'connectivity_normalized_log2(n)',
                'connectivity_normalized_sqrt(n)/5',
                'conductance',
            ],
        )

    df.to_csv(outfile, index=False)
    print("Done")

    if len(universal_before) > 0:
        print("Writing extra outputs from CM2Universal")

        cluster_sizes = {
            key.replace('"', ''): val
            for key, val in zip(ids, ns)
        }
        output_entries = []
        with open(universal_before, encoding="utf-8") as json_file:
            before = json.load(json_file)
            for cluster in before:
                if not cluster['extant']:
                    output_entries.append({
                        "input_cluster": cluster['label'],
                        'n': len(cluster['nodes']),
                        'extant': False,
                        'descendants': {
                            desc: cluster_sizes[desc]
                            for desc in cluster['descendants']
                            if desc in cluster_sizes
                        }
                    })
                else:
                    output_entries.append({
                        "input_cluster": cluster['label'],
                        'n': len(cluster['nodes']),
                        'extant': True
                    })

        # Specify the file path for the JSON output
        json_file_path = outfile + '_to_universal.json'
        csv_file_path = outfile + '_to_universal.csv'

        # Get lines for the csv format
        csv_lines = ['input_cluster,n,descendant,desc_n,extant']
        for entry in output_entries:
            if entry['extant']:
                csv_lines.append(f'{entry["input_cluster"]},{entry["n"]},,,1')
            elif len(entry['descendants']) == 0:
                csv_lines.append(f'{entry["input_cluster"]},{entry["n"]},,,0')
            else:
                for descendant, desc_n in entry['descendants'].items():
                    csv_line_entries = map(
                        str,
                        [
                            entry["input_cluster"],
                            entry["n"],
                            descendant,
                            desc_n,
                            0,
                        ],
                    )
                    csv_line = ','.join(csv_line_entries)
                    csv_lines.append(csv_line)

        print("\tWriting JSON")
        # Write the array of dictionaries as formatted JSON to the file
        with open(json_file_path, 'w', encoding="utf-8") as json_file:
            json.dump(output_entries, json_file, indent=4)
        print("\tDone")

        print("\tWriting CSV")
        # Write the lines to the file
        with open(csv_file_path, 'w', encoding="utf-8") as file:
            for line in csv_lines:
                file.write(line + '\n')
        print("\tDone")
        print("Done")


def entry_point():
    '''
    entry point, calls the main function
    '''
    typer.run(main)


if __name__ == "__main__":
    entry_point()
