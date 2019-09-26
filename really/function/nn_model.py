'''
Created on 1 Mar 2019

@author: enerve
'''

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from .flatten import Flatten

import logging
from really import util

class NNModel():
    '''
    Manages and trains a neural-network
    '''

    def __init__(self,
                 criterion_str,
                 optimizer_str,
                 alpha,
                 regularization_param,
                 batch_size,
                 max_iterations):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Using NN model")

        self.optimizer_str = optimizer_str
        self.alpha = alpha
        self.regularization_param = regularization_param
        self.criterion_str = criterion_str
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')

        # Stats / debugging
        self.stat_error_cost = []
        self.stat_reg_cost = []
        self.stat_val_error_cost = []
                
        #self.sids = self._sample_ids(3000, self.batch_size)
        #self.last_loss = torch.zeros(self.batch_size, 7).cuda()
        
    def prefix(self):
        return 'neural_a%s_r%s_b%d_i%d_C%s_O%s' % (self.alpha, 
                                     self.regularization_param,
                                     self.batch_size,
                                     self.max_iterations,
                                     self.criterion_str,
                                     self.optimizer_str)

    def init_net(self, net):        
        net.to(self.device)

        self.net = net
        self.logger.debug("Net:\n%s", self.net)

        if self.criterion_str == 'mse':
            self.criterion = nn.MSELoss(reduce=False)
        elif self.criterion_str == 'bce':
            self.criterion = nn.BCELoss(reduce=False)
        else:
            self.logger.error("Unspecified NN Criterion")
        
        if self.optimizer_str == 'sgd':
            self.optimizer = optim.SGD(
                net.parameters(),
                lr=self.alpha,
                momentum=0.9,
                dampening=0,
                weight_decay=self.regularization_param,
                nesterov=False)
        elif self.optimizer_str == 'adam':
            self.optimizer = optim.Adam(
                net.parameters(),
                lr=self.alpha,
                weight_decay=self.regularization_param,
                amsgrad=False)
        else:
            self.logger.error("Unspecified NN Optimizer")
            
        #self.num_outputs = 

    def activations(self, Xbatch):
        self.net.eval()
        with torch.no_grad():
            output, activations = self.net(Xbatch)
        return activations

    def value(self, Xbatch):
        self.net.eval()
        with torch.no_grad():
            output, _ = self.net(Xbatch)
        return output
        

    def _sample_ids(self, l, n):
        ids = torch.cuda.FloatTensor(n) if util.use_gpu else torch.FloatTensor(n)
        ids = (l * ids.uniform_()).long()
        return ids

    def train(self, SHX, SHT, SHM, VSHX, VSHT, VSHM):
        SHX = SHX.to(self.device)
        SHT = SHT.to(self.device)
        SHM = SHM.to(self.device)
        VSHX = VSHX.to(self.device)
        VSHT = VSHT.to(self.device)
        VSHM = VSHM.to(self.device)
        
        N = SHX.shape[0]
        self.logger.debug("Training with %d items...", N)


        # for stats
        preferred_samples = 100
        period = self.max_iterations // preferred_samples
        period = max(period, 10)
        if period > self.max_iterations:
            self.logger.warning("max_iterations too small for period plotting")

        sum_error_cost = None

        for i in range(self.max_iterations):
            self.optimizer.zero_grad()
            
            if self.batch_size == 0:
                # Do full-batch
                X = SHX   # N x di
                Y = SHT   # N
                M = SHM   # N x do
            else:
                ids = self._sample_ids(N, self.batch_size)
                X = SHX[ids]   # b x di
                Y = SHT[ids]   # b
                M = SHM[ids]   # b x do
            Y = torch.unsqueeze(Y, 1)   # b x 1
            
            # forward
            self.net.train() # Set Training mode
            outputs, _ = self.net(X)       # b x do
            
            # loss
            Y = Y * M  # b x do
            loss = self.criterion(outputs, Y)  # b x do
            with torch.no_grad():
                # Zero-out the computed losses for the other actions/outputs
                loss *= M   # b x do
            # backward
            onez = torch.ones(loss.shape).to(self.device) #TODO: move out?
            
            
            loss.backward(onez)
            
            # updated weights
            self.optimizer.step()
            
            # Stats
            with torch.no_grad():
                suml = torch.sum(loss, 0)
                countl = torch.sum(loss > 0, 0).float()

#                 ltz = (suml < 0).byte()
#                 if ltz.any():
#                     self.logger.debug("loss < 0")
#                 
#                 if i==0:
#                     self.logger.debug("Initial loss:\n  %s", suml / (countl + 0.0001))
#                 if i+1==self.max_iterations:
#                     self.logger.debug("   Last loss:\n  %s", suml / (countl + 0.0001))

                if sum_error_cost is None:
                    sum_error_cost = suml.clone().to(self.device)
                    sum_error_cost.detach()
                    count_actions = countl.clone().to(self.device)
                else:
                    sum_error_cost.add_(suml)  # do
                    count_actions.add_(countl)  # do

                if (i+1) % period == 0:
                    mean_error_cost = sum_error_cost / (count_actions + 0.01)
                    self.stat_error_cost.append(mean_error_cost.cpu().numpy())
    
                    #self.logger.debug("  loss=%0.2f", sum_error_cost.mean().item())
    
                    torch.zeros(sum_error_cost.shape[0], out=sum_error_cost)
                    torch.zeros(count_actions.shape[0], out=count_actions)
                    
                    # Validation
                    self.net.eval()
                    X = VSHX
                    Y = torch.unsqueeze(VSHT, 1)
                    M = VSHM   # N x do
                    outputs, _ = self.net(X)       # b x do
                    Y = Y * M  # b x do
                    loss = self.criterion(outputs, Y)  # b x do
                    loss *= M   # b x do
                    
                    suml = torch.sum(loss, 0)
                    countl = torch.sum(loss > 0, 0).float()
                    mean_error_cost = suml / (countl + 0.01)
                    self.stat_val_error_cost.append(mean_error_cost.cpu().numpy())

                    #for param in self.net.parameters():
                    #    self.logger.debug("  W=%0.6f dW=%0.6f    %s", 
                    #                      torch.mean(torch.abs(param.data)),
                    #                      torch.mean(torch.abs(param.grad)),
                    #                      param.shape)

                    #self.live_stats()

            if (i+1) % 1000 == 0:
                self.logger.debug("   %d / %d", i+1, self.max_iterations)

        self.logger.debug("  trained \tN=%s \tE=%0.3f \tVE=%0.3f", N,
                          self.stat_error_cost[-1].mean().item(),
                          self.stat_val_error_cost[-1].mean().item())

    def test(self):
        pass

    def collect_stats(self, ep):
        pass
    
    def collect_epoch_stats(self, epoch):
        pass
    
    def save_stats(self, pref=""):
        # TODO
        pass

    def load_stats(self, subdir, pref=""):
        # TODO
        pass

    def report_stats(self, pref=""):
        num = len(self.stat_error_cost[1:])

        n_cost = np.asarray(self.stat_error_cost[1:]).T
        labels = list(range(n_cost.shape[1]))        

        n_v_cost = np.asarray(self.stat_val_error_cost[1:]).T
        labels.extend(["val%d" % i for i in range(n_v_cost.shape[1])])
        cost = np.concatenate([n_cost, n_v_cost], axis=0)
        avgcost = np.stack([n_cost.mean(axis=0), n_v_cost.mean(axis=0)], axis=0)
        
        util.plot(cost,
                  range(num),
                  labels = labels,
                  title = "NN training/validation cost across actions",
                  pref=pref+"cost",
                  ylim=None)

        util.plot(avgcost,
                  range(num),
                  labels = ["training cost", "validation cost"],
                  title = "NN training/validation cost",
                  pref=pref+"avgcost",
                  ylim=None)

    
    def live_stats(self):
        num = len(self.stat_error_cost[1:])
        
        if num < 1:
            return

        n_cost = np.asarray(self.stat_error_cost[1:]).T
        n_v_cost = np.asarray(self.stat_val_error_cost[1:]).T
        avgcost =  np.stack([n_cost.mean(axis=0), n_v_cost.mean(axis=0)], axis=0)

        util.plot(avgcost,
                  range(num),
                  labels = ["training cost", "validation cost"],
                  title = "NN training/validation cost",
                  live=True)
        
    def save_model(self, pref=""):
        self.logger.debug("Saving model")
        util.torch_save(self.net, "NN_" + pref)
        self.logger.debug("Saving model state dict")
        util.torch_save(self.net.state_dict(), "NN_sd_" + pref)

    def load_model(self, load_subdir, pref=""):
        fname = "NN_" + pref
        self.logger.debug("Loading model %s", fname)
        net = util.torch_load(fname, load_subdir)
        net.eval()
        self.init_net(net)
