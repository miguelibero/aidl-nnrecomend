import sys
from nnrecommend.cli.main import main, Context
from nnrecommend.cli.graph import dataset_graph, model_graph
from nnrecommend.cli.train import train
from nnrecommend.cli.fit import fit


main.add_command(train)
main.add_command(fit)
main.add_command(dataset_graph)
main.add_command(model_graph)


if __name__ == "__main__":
    sys.exit(main(obj=Context()))