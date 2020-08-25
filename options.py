from argparse import ArgumentParser
import os
import shutil

parser = ArgumentParser(add_help=False)

# train or test stage setting
parser.add_argument('--phase', type=str, default='train',
                    choices=['train', 'test'])

parser.add_argument('--model_name', type=str, default='AdaIN')

parser.add_argument('--save_root', type=str, default='./save/')
parser.add_argument('--weight_root', type=str, default='weights')
parser.add_argument('--logger_root', type=str, default='loggers')
parser.add_argument('--tblogs_root', type=str, default='tblogs')
parser.add_argument('--train_result_root', type=str, default='results')

parser.add_argument('--test_result_root', type=str, default='test_results')

parser.add_argument('--content_dir', type=str, default='G:/Transfer/content/images/')
parser.add_argument('--style_dir', type=str, default='G:/Transfer/style/Degas/')

parser.add_argument('--continue_train', action="store_true", help="optional")
parser.add_argument('--use_gpu', action="store_true", help="optional")
parser.add_argument('--aug', action="store_true", help="optional")

parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)

parser.add_argument('--epochs', type=int, default=16000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--log_interval', type=int, default=10)

hparams = parser.parse_args()

def check_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def check_paras(hparams):
    os.makedirs(hparams.save_root, exist_ok=True)

    weight_root = os.path.join(hparams.save_root, 'weights', hparams.model_name)
    check_dir(weight_root)
    hparams.weight_root = weight_root

    logger_root = os.path.join(hparams.save_root, 'loggers', hparams.model_name)
    check_dir(logger_root)
    hparams.logger_root = logger_root

    tblogs_root = os.path.join(hparams.save_root, 'tblogs', hparams.model_name)
    check_dir(tblogs_root)
    hparams.tblogs_root = tblogs_root

    train_result_root = os.path.join(hparams.save_root, 'results', hparams.model_name)
    check_dir(train_result_root)
    hparams.train_result_root = train_result_root

    return hparams

hparams = check_paras(hparams)

if __name__ == '__main__':
    print(str(hparams))
