import sys
from nnrecommend.cli.main import main, Context
from nnrecommend.cli.explore import explore_dataset, explore_model
from nnrecommend.cli.train import train
from nnrecommend.cli.fit import fit
from nnrecommend.cli.tune import tune
from nnrecommend.cli.recommend import recommend


main.add_command(train)
main.add_command(fit)
main.add_command(tune)
main.add_command(explore_dataset)
main.add_command(explore_model)
main.add_command(recommend)


if __name__ == "__main__":
    sys.exit(main(obj=Context()))