import os, inspect, time

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def makedir(path):

    try: os.mkdir(path)
    except: pass

def save_graph(contents, xlabel, ylabel, savename):

    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()

def psnr(input, target):

    psnr = torch.log(1 / torch.sqrt(torch.mean((target - input)**2))) / np.log(10.0) * 20
    return psnr

def torch2npy(input):

    output = input.detach().numpy()
    return output

def training(neuralnet, data_loader, epochs, batch_size):

    start_time = time.time()
    loss_tr = 0
    list_loss = []
    list_psnr = []

    makedir(PACK_PATH+"/training")
    makedir(PACK_PATH+"/static")
    makedir(PACK_PATH+"/static/reconstruction")

    print("\nTraining SRCNN to %d epochs" %(epochs))

    for epoch in range(epochs):

        neuralnet.model.train()
        running_loss, running_psnr = 0, 0

        for i, (data, label) in enumerate(data_loader):
            neuralnet.optimizer.zero_grad()
            data, label = data.cuda(), label.cuda()

            output = neuralnet.model(data)
            loss = neuralnet.mse(output, label)

            running_loss += loss.item()
            running_psnr += psnr(output, label).item()

            loss.backward()
            neuralnet.optimizer.step()

            del data, label, output, loss
            torch.cuda.empty_cache()

        loss_tr = running_loss / len(data_loader)
        psnr_tr = running_psnr / len(data_loader)
        list_loss.append(loss_tr)
        list_psnr.append(psnr_tr)

        print("Epoch [%d / %d] | Loss: %f  PSNR: %f" %(epoch, epochs, loss_tr, psnr_tr))
        torch.save(neuralnet.model.state_dict(), PACK_PATH+"/runs/params")

    print("Final Epcoh | Loss: %f  PSNR: %f" %(loss_tr, psnr_tr))

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    save_graph(contents=list_loss, xlabel="Iteration", ylabel="L2 loss", savename="loss")
    save_graph(contents=list_psnr, xlabel="Iteration", ylabel="PSNR (dB)", savename="psnr")

def validation(neuralnet, data_loader):

    if(os.path.exists(PACK_PATH+"/runs/params")):
        neuralnet.model.load_state_dict(torch.load(PACK_PATH+"/runs/params"))
        neuralnet.model.eval()

    makedir(PACK_PATH+"/test")
    makedir(PACK_PATH+"/test/reconstruction")

    start_time = time.time()
    print("\nValidation")
    neuralnet.model.eval()
    running_loss, running_psnr = 0, 0

    for i, (data, label) in enumerate(data_loader):
        data, label = data.cuda(), label.cuda()

        output = neuralnet.model(data)
        loss = neuralnet.mse(output, label)

        running_loss += loss.item()
        running_psnr += psnr(output, label).item()

        del data, label, output, loss
        torch.cuda.empty_cache()
    loss_tr = running_loss / len(data_loader)
    psnr_tr = running_psnr / len(data_loader)

    elapsed_time = time.time() - start_time
    print("Final validation | Loss: %f  PSNR: %f" %(loss_tr, psnr_tr))
    print("Elapsed: "+str(elapsed_time))
