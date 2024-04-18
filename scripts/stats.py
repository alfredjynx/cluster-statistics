# pylint: disable=c-extension-no-member,fixme
'''
Defining the class that computes cluster statistics
'''

import json
import os

import networkit as nk
import pandas as pd
import typer
from cluster_statistics.stats import from_existing_clustering
from hm01.graph import Graph
from hm01.mincut import viecut
from numpy import log2, log10


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
