import argparse
import yaml

from utils.supervisor import Supervisor


def main(args) :

    with open(args.config_filename, 'r', encoding="utf-8") as f :

        supervisor_config = yaml.safe_load(f)

        supervisor = Supervisor(**supervisor_config)  # !!!

        # 测试
        test_loss, test_results = supervisor.evaluate(dataset='test')

        base_message = ''

        # 输出评价指标：MAE、RMSE、MAPE
        supervisor.show_metrics(test_results['prediction'], test_results['truth'], base_message, 0.0)


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default="", type=str, 
                        help='Configuration filename for restoring the model.')
    args = parser.parse_args()
    main(args)

