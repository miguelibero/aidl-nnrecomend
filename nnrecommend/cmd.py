import click
from typing import Tuple, Dict
from nnrecommend.logging import setup_log


@click.command('recommender system using deep learning')
@click.option('-v', '--verbose', type=bool, is_flag=True, help='print verbose output')
@click.option('--logoutput', type=str, help='append output to this file')
def main(verbose: bool, logoutput: str,):
    logger = setup_log(verbose, logoutput)
    logger.info("working!")


if __name__ == "__main__":
    sys.exit(main())