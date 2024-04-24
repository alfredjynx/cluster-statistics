# pylint: disable=c-extension-no-member,fixme
'''
Defining the class that computes cluster statistics
'''

import typer
from cluster_statistics.stats import Statistics


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
        Statistics Script
        Utilizes the Class
    '''

    stats = Statistics(input_file, existing_clustering, universal_before,
                       resolution, output)

    stats.from_tsv()

    stats.compute_stats()

    stats.to_csv()



def entry_point():
    '''
    entry point, calls the main function
    '''
    typer.run(main)


if __name__ == "__main__":
    entry_point()
