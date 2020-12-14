import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch

import source.neuralnet as nn
from source.datamanager import load_data
import source.solver as solver

def main():

    if(not(torch.cuda.is_available())): FLAGS.ngpu = 0
    device = torch.device("cuda" if (torch.cuda.is_available() and FLAGS.ngpu > 0) else "cpu")
    bicubic = FLAGS.bicubic==1
    
    srnet = nn.NeuralNet(device=device, ngpu=FLAGS.ngpu, model=FLAGS.model, bicubic=bicubic)

    train_loader = load_data(bicubic=bicubic)
    test_loader = load_data(train=False, bicubic=bicubic)

    solver.training(neuralnet=srnet, data_loader=train_loader, test_loader=test_loader, epochs=FLAGS.epoch, batch_size=FLAGS.batch)
    solver.validation(neuralnet=srnet, data_loader=test_loader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='-')
    parser.add_argument('--epoch', type=int, default=5000, help='-')
    parser.add_argument('--batch', type=int, default=16, help='-')
    parser.add_argument('--model', type=str, default='SRCNN', help='-')
    parser.add_argument('--bicubic', type=int, default=1, help='-')
    FLAGS, unparsed = parser.parse_known_args()

    main()
