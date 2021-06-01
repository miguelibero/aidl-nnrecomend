import sys
from nnrecommend.cli.main import main, Context
from nnrecommend.cli.explore import explore_dataset, explore_model
from nnrecommend.cli.train import train
from nnrecommend.cli.fit import fit


main.add_command(train)
main.add_command(fit)
main.add_command(explore_dataset)
main.add_command(explore_model)


if __name__ == "__main__":
    sys.exit(main(obj=Context()))