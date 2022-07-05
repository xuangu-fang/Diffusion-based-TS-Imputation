import numpy as np
import torch
from torch.optim import Adam
# from sklearn.cluster import KMeans
from kernels import KernelRBF
import random
import tqdm

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

class GPTF_time:
    #Uinit: init U
    #m: #pseudo inputs
    #ind: entry indices
    #y: observed tensor entries
    #B: batch-size
    def __init__(self, ind, y, time_points, Uinit, m, B, device, jitter=1e-4):
        self.device = device
        self.Uinit = Uinit
        self.m = m
        self.y = torch.tensor(y.reshape([y.size,1]), device=self.device)
        self.ind = ind
        self.time_points = torch.tensor(time_points.reshape([time_points.size,1]), device=self.device)
        self.B = B
        self.nmod = len(self.Uinit)
        self.U = [torch.tensor(self.Uinit[k].copy(), device=self.device, requires_grad=True) for k in range(self.nmod)]
        #dim. of pseudo input
        self.d = 1
        for k in range(self.nmod):
            self.d = self.d + Uinit[k].shape[1]
        #init mu, L, Z
        #Zinit = self.init_pseudo_inputs()
        Zinit = np.random.rand(self.m, self.d)
        self.Z = torch.tensor(Zinit, device=self.device, requires_grad=True)
        self.N = y.size
        #variational posterior
        self.mu = torch.tensor(np.zeros([m,1]), device=self.device, requires_grad=True)
        self.L = torch.tensor(np.eye(m), device=self.device, requires_grad=True)
        #kernel parameters
        #self.log_amp = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log_ls = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log_tau = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.jitter = torch.tensor(jitter, device=self.device)
        self.kernel = KernelRBF(self.jitter)
        
        
    #batch neg ELBO
    def nELBO_batch(self, sub_ind):
        input_emb = torch.cat([self.U[k][self.ind[sub_ind, k],:] for k in range(self.nmod)], 1)
        #time_log = torch.log(self.time_points[sub_ind] + 1e-4)
        #input_emb = torch.cat([input_emb, time_log], 1)
        input_emb = torch.cat([input_emb, self.time_points[sub_ind]], 1)
        y_sub = self.y[sub_ind]
        Kmm = self.kernel.matrix(self.Z, torch.exp(self.log_ls))
        Kmn = self.kernel.cross(self.Z, input_emb, torch.exp(self.log_ls))
        Knm = Kmn.T
        Ltril = torch.tril(self.L)
        KnmKmmInv = torch.linalg.solve(Kmm, Kmn).T
        KnmKmmInvL = torch.matmul(KnmKmmInv, Ltril)
        tau = torch.exp(self.log_tau)
        ls = torch.exp(self.log_ls)

        hh_expt = torch.matmul(Ltril, Ltril.T) + torch.matmul(self.mu, self.mu.T)
        ELBO = -0.5*torch.logdet(Kmm) - 0.5*torch.trace(torch.linalg.solve(Kmm, hh_expt)) + 0.5*torch.sum(torch.log(torch.square(torch.diag(Ltril)))) \
                + 0.5*self.N*self.log_tau - 0.5*tau*self.N/self.B*torch.sum(torch.square(y_sub - torch.matmul(KnmKmmInv, self.mu))) \
                - 0.5*tau*( self.N*(1.0+self.jitter) - self.N/self.B*torch.sum(KnmKmmInv*Knm) + self.N/self.B*torch.sum(torch.square(KnmKmmInvL)) ) \
                + 0.5*self.m - 0.5*self.N*torch.log(2.0*torch.tensor(np.pi, device=self.device))

        return -torch.squeeze(ELBO)

    # def init_pseudo_inputs(self):
    #     part = [None for k in range(self.nmod)]
    #     for k in range(self.nmod):
    #         part[k] = self.Uinit[k][self.ind[:,k], :]
    #     X = np.hstack(part)

    #     X = X[np.random.randint(X.shape[0], size=self.m * 100), :]
    #     print(X.shape)

    #     kmeans = KMeans(n_clusters=self.m, random_state=0).fit(X)
    #     return kmeans.cluster_centers_


    def pred(self, test_ind, test_time):
        inputs = torch.cat([self.U[k][test_ind[:,k],:]  for k in range(self.nmod)], 1)
        inputs = torch.cat([inputs, test_time],1)
        #test_time_log = torch.log(test_time + 1e-4)
        #inputs = torch.cat([inputs, test_time_log],1)
        Knm = self.kernel.cross(inputs, self.Z, torch.exp(self.log_ls))
        Kmm = self.kernel.matrix(self.Z, torch.exp(self.log_ls))
        pred_mean = torch.matmul(Knm, torch.linalg.solve(Kmm, self.mu))
        return pred_mean




    def _callback(self, ind_te, yte, time_te):
        with torch.no_grad():

            MSE_loss = torch.nn.MSELoss()
            MAE_loss = torch.nn.L1Loss()

            ls = torch.exp(self.log_ls)
            tau = torch.exp(self.log_tau)
            pred_mean = self.pred(ind_te, time_te)
            #err_tr = torch.sqrt(torch.mean(torch.square(pred_mu_tr-ytr)))
            pred_tr = self.pred(self.ind, self.time_points)
            err_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))
            err_te_rmse = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))
            err_te_MAE = MAE_loss(pred_mean.squeeze(),yte.squeeze())

            print('ls=%.5f, tau=%.5f, train_err = %.5f, test_err=%.5f' %\
                 (ls, tau, err_tr, err_te_rmse))
            # with open('sparse_gptf_res.txt','a') as f:
            #     f.write('%g '%err_te)
            return err_te_rmse,err_te_MAE
    
    def train(self, ind_te, yte, time_te, lr, max_epochs=100):
        #yte = torch.tensor(yte.reshape([yte.size,1]), device=self.device)

        yte = yte.reshape([-1, 1]).to(self.device)
        #time_te = torch.tensor(time_te.reshape([time_te.size, 1]), device=self.device)
        time_te = time_te.reshape([-1,1]).to(self.device)


        paras = self.U + [self.Z, self.mu, self.L, self.log_ls, self.log_tau]
        
        minimizer = Adam(paras, lr=lr)
        for epoch in tqdm.tqdm(range(max_epochs)):
            curr = 0
            while curr < self.N:
                batch_ind = np.random.choice(self.N, self.B, replace=False)
                minimizer.zero_grad()
                loss = self.nELBO_batch(batch_ind)
                loss.backward(retain_graph=True)
                minimizer.step()
                curr = curr + self.B
            print('epoch %d done'%epoch)
            if epoch%5 == 0:
                self._callback(ind_te, yte, time_te)
                
        return self._callback(ind_te, yte, time_te)


class GPTF:
    #Uinit: init U
    #m: #pseudo inputs
    #ind: entry indices
    #y: observed tensor entries
    #B: batch-size
    def __init__(self, ind, y, Uinit, m, B, device, jitter=1e-3):
        self.device = device
        self.Uinit = Uinit
        self.m = m
        self.y = torch.tensor(y.reshape([y.size,1]), device=self.device)
        self.ind = ind
        self.B = B
        self.nmod = len(self.Uinit)
        self.U = [torch.tensor(self.Uinit[k].copy(), device=self.device, requires_grad=True) for k in range(self.nmod)]
        #dim. of pseudo input
        self.d = 0
        for k in range(self.nmod):
            self.d = self.d + Uinit[k].shape[1]
        #init mu, L, Z
        #Zinit = self.init_pseudo_inputs()
        Zinit = np.random.rand(self.m, self.d)
        self.Z = torch.tensor(Zinit, device=self.device, requires_grad=True)
        self.N = y.size
        #variational posterior
        self.mu = torch.tensor(np.zeros([m,1]), device=self.device, requires_grad=True)
        self.L = torch.tensor(np.eye(m), device=self.device, requires_grad=True)
        #kernel parameters
        #self.log_amp = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log_ls = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log_tau = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.jitter = torch.tensor(jitter, device=self.device)
        self.kernel = KernelRBF(self.jitter)

        print('N =',self.N)
        
        
    #batch neg ELBO
    def nELBO_batch(self, sub_ind):
        input_emb = torch.cat([self.U[k][self.ind[sub_ind, k],:] for k in range(self.nmod)], 1)
        y_sub = self.y[sub_ind]
        Kmm = self.kernel.matrix(self.Z, torch.exp(self.log_ls))
        Kmn = self.kernel.cross(self.Z, input_emb, torch.exp(self.log_ls))
        Knm = Kmn.T
        Ltril = torch.tril(self.L)
        KnmKmmInv = torch.linalg.solve(Kmm, Kmn).T
        KnmKmmInvL = torch.matmul(KnmKmmInv, Ltril)
        tau = torch.exp(self.log_tau)
        ls = torch.exp(self.log_ls)

        hh_expt = torch.matmul(Ltril, Ltril.T) + torch.matmul(self.mu, self.mu.T)
        ELBO = -0.5*torch.logdet(Kmm) - 0.5*torch.trace(torch.linalg.solve(Kmm, hh_expt)) + 0.5*torch.sum(torch.log(torch.square(torch.diag(Ltril)))) \
                + 0.5*self.N*self.log_tau - 0.5*tau*self.N/self.B*torch.sum(torch.square(y_sub - torch.matmul(KnmKmmInv, self.mu))) \
                - 0.5*tau*( self.N*(1.0+self.jitter) - self.N/self.B*torch.sum(KnmKmmInv*Knm) + self.N/self.B*torch.sum(torch.square(KnmKmmInvL)) ) \
                + 0.5*self.m - 0.5*self.N*torch.log(2.0*torch.tensor(np.pi, device=self.device))

        return -torch.squeeze(ELBO)


    def pred(self, test_ind):
        inputs = torch.cat([self.U[k][test_ind[:,k],:]  for k in range(self.nmod)], 1)
        Knm = self.kernel.cross(inputs, self.Z, torch.exp(self.log_ls))
        Kmm = self.kernel.matrix(self.Z, torch.exp(self.log_ls))
        pred_mean = torch.matmul(Knm, torch.linalg.solve(Kmm, self.mu))
        return pred_mean


    def _callback(self, ind_te, yte):
        with torch.no_grad():
            MSE_loss = torch.nn.MSELoss()
            MAE_loss = torch.nn.L1Loss()

            ls = torch.exp(self.log_ls)
            tau = torch.exp(self.log_tau)
            pred_mean = self.pred(ind_te)
            #err_tr = torch.sqrt(torch.mean(torch.square(pred_mu_tr-ytr)))
            pred_tr = self.pred(self.ind)
            err_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))
            err_te_rmse = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))
            err_te_MAE = MAE_loss(pred_mean, yte)
            print('ls=%.5f, tau=%.5f, train_err = %.5f, test_err=%.5f' %\
                 (ls, tau, err_tr, err_te_rmse))
            # with open('sparse_gptf_res.txt','a') as f:
            #     f.write('%g '%err_te)
            return err_te_rmse,err_te_MAE
    
    def train(self, ind_te, yte, lr, max_epochs=3):
        # yte = torch.tensor(yte.reshape([yte.size,1]), device=self.device)
        yte = yte.reshape([-1, 1]).to(self.device)
        paras = self.U + [self.Z, self.mu, self.L, self.log_ls, self.log_tau]
        
        minimizer = Adam(paras, lr=lr)
        for epoch in range(max_epochs):
            curr = 0
            while curr < self.N:
                batch_ind = np.random.choice(self.N, self.B, replace=False)
                minimizer.zero_grad()
                loss = self.nELBO_batch(batch_ind)
                # print('loss=',loss)
                loss.backward(retain_graph=True)
                minimizer.step()
                curr = curr + self.B
                # print("N=%d,B=%d,process=%.3f"%(self.N,self.B,curr/self.N))
            print('epoch %d done'%epoch)
            if epoch%2 == 0:
                self._callback(ind_te, yte)
                
        return self._callback(ind_te, yte)