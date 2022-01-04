import argparse
import json
import logging
import os
import torch
import scipy.optimize as opt
import numpy as np
from data import DBP15K_new
from model import DualConsensusNet


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing dual-consensus Models',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--category', default='zh_en', type=str)
    parser.add_argument('-save', '--save_path', default='./save/', type=str)
    parser.add_argument('--p', type=float, default=0.3)
    parser.add_argument('--dim', type=int, default=256)
    # parser.add_argument('--rnd_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--inshape', type=int, default=300)
    parser.add_argument('--embedding_shape', type=int, default=300)
    parser.add_argument('--sinknum', type=int, default=10)
    parser.add_argument('--color_size', type=int, default=50)
    parser.add_argument('--begin_steps', type=int, default=100) #default  =  100
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--negative_number', type=int, default=10)
    # negative_number

    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU')
    parser.add_argument('--device', type=str, default='cuda:1', help='') #cuda:1
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_valid', action='store_true', default=True)
    parser.add_argument('--do_test', action='store_true', default=True)

    parser.add_argument('--evaluate_train', action='store_true', default = False, help='Evaluate on training data')
    parser.add_argument('--data_path', type=str, default='./data/DBP15K')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('--max_steps', default=350, type=int)
    # parser.add_argument('--change_steps', default=100000, type=int)
    # parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--save_checkpoint_steps', default=100, type=int)
    parser.add_argument('--valid_steps', default= 20, type=int)
    parser.add_argument('--log_steps', default=50, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=20, type=int, help='valid/test log every xx steps')
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint.pth')
    )


def hungarian(s: torch.Tensor, n1=None, n2=None):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :return: optimal permutation matrix
    """
    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    for b in range(batch_num):
        n1b = perm_mat.shape[1] if n1 is None else n1[b]
        n2b = perm_mat.shape[2] if n2 is None else n2[b]
        row, col = opt.linear_sum_assignment(perm_mat[b, :n1b, :n2b])
        perm_mat[b] = np.zeros_like(perm_mat[b])
        perm_mat[b, row, col] = 1
    perm_mat = torch.from_numpy(perm_mat).to(device)

    return perm_mat
def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)
    device = torch.device(args.device)

    # logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    data = DBP15K_new(args)
    model = DualConsensusNet(args)

    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        model = model.to(device)

    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint.pth'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing Dual_net Model...')
        init_step = 0

    step = init_step
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)


    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []

        # Training Loop

        for step in range(init_step, args.max_steps):

            log = model.train_step(model, optimizer, data, args)
            training_logs.append(log)
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    # 'warm_up_steps': warm_up_steps,
                    # 'change_steps': change_steps
                }
                save_model(model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and (step + 1) % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = model.test_step(model, data, args, 'test')
                log_metrics('Valid', step, metrics)

            if args.evaluate_train and (step+1) % args.valid_steps == 0:
                logging.info('Evaluating on Training Dataset...')
                metrics = model.test_step(model, data, args, 'train')
                log_metrics('Train', step, metrics)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            # 'warm_up_steps': warm_up_steps,
            # 'change_steps': change_steps
        }
        save_model(model, optimizer, save_variable_list, args)

    # if args.do_valid:
    #     logging.info('Evaluating on Valid Dataset...')
    #     metrics = rule_model.test_step(rule_model, valid_triples, all_true_triples, args)
    #     log_metrics('Valid', step, metrics)
    #
    # if args.do_test:
    #     logging.info('Evaluating on Test Dataset...')
    #     metrics = rule_model.test_step(rule_model, test_triples, all_true_triples, args)
    #     log_metrics('Test', step, metrics)
    #
    # if args.evaluate_train:
    #     logging.info('Evaluating on Training Dataset...')
    #     metrics = rule_model.test_step(rule_model, train_triples, all_true_triples, args)
    #     log_metrics('Test', step, metrics)


if __name__ == '__main__':
    main(parse_args())