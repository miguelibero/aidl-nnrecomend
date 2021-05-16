import sys
from nnrecommend.cli.main import main, Context
from nnrecommend.cli.graph import graph
from nnrecommend.cli.train import train


main.add_command(graph)
main.add_command(train)


if __name__ == "__main__":
    sys.exit(main(obj=Context()))