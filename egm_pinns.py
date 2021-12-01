import deepxde as dde
import fenton as fk
import h5py
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os

class fk_system():
    
    def __init__(self, min_t, max_t, t_step = 1):
    
        self.min_x = 0
        self.max_x = 4
        self.min_y = 0
        self.max_y = 4
        self.min_t = min_t # in milliseconds
        self.max_t = max_t
        self.t_step = t_step
        self.geom = dde.geometry.Rectangle([self.min_x,self.min_y], [self.max_x,self.max_y])
        self.timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.timedomain)


    def check_simulation(self, filename, min_t, max_t, t_step, rows=1, columns=5, egm = False):
        self.params, self.D = fk.io.load_params(filename)
        for key in self.params:
            self.params[key] = torch.as_tensor(self.params[key])
        
        
        f = h5py.File(filename, 'r')
        data = f['states'][min_t:max_t:t_step, 2 + 2*int(egm)]
        t_idx = np.linspace(0, len(data)-1, rows*columns, dtype = int)
        fig = plt.figure(figsize = (columns*10, rows*10))
        c = 0
        while c < len(t_idx):
            ax = plt.subplot(rows, columns, c+1)
            im = ax.imshow(data[t_idx[c]])
            plt.colorbar(im, ax = ax)
            c+=1
        plt.show()
        


    def read_simulation(self, filename):
        self.params, self.D = fk.io.load_params(filename)
        for key in self.params:
            self.params[key] = torch.as_tensor(self.params[key])
        
        self.dx = self.params['dx']/1 # to turn array into float
        self.dt = self.params['dt']/1
        X = np.linspace(0, 4, 128)
        Y = np.linspace(0, 4, 128)
        T = np.arange(self.min_t, self.max_t, self.t_step)
        t, x, y = np.meshgrid(T, X, Y, indexing = 'ij')
        x = x.reshape(-1,1)
        t = t.reshape(-1,1)
        y = y.reshape(-1,1)

        f = h5py.File(filename, 'r')

        v = f['states'][self.min_t: self.max_t:self.t_step, 0]#.transpose((1,2,0))
        w = f['states'][self.min_t: self.max_t:self.t_step, 1]#.transpose((1,2,0))
        u = f['states'][self.min_t: self.max_t:self.t_step, 2]#.transpose((1,2,0))
        phi = f['states'][self.min_t: self.max_t:self.t_step, 4]#.transpose((1,2,0))
        

        v = v.reshape(-1,1)
        u = u.reshape(-1,1)
        w = w.reshape(-1,1)
        phi = phi.reshape(-1,1)
        d = np.tile(self.D, [len(T), 1, 1])#.transpose(1,2,0).reshape(-1,1)
#         d = self.D.reshape(-1,1)
#         x_d, y_d = np.meshgrid(X, Y)
#         x_d.reshape(-1,1)
#         y_d.reshape(-1,1)
        d = d.reshape(-1,1)
        
        return np.hstack((x, y, t)), v, w, u, d, phi # np.hstack((x,y)), d


    
    
    def  homogeneous_pde_loss(self, x, u, d):
        """
        x = [x,y,t]
        y = [v, w, u, DD]
        """

        v = u[:,0]
        w = u[:,1]
        u = u[:,2]


        du = torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        dw = torch.autograd.grad(w, x, grad_outputs = torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        dv = torch.autograd.grad(v, x, grad_outputs = torch.ones_like(u), retain_graph=True, create_graph=True)[0]

        du_t, du_x, du_y = du[:,2], du[:,0], du[:,1]
        dv_t = dv[:,2]
        dw_t = dw[:,2]

        # gate variables
        p = torch.gt(u, self.params["V_c"]).int()
        q = torch.gt(u, self.params["V_v"]).int()
        tau_v_minus = (1 - q) * self.params["tau_v1_minus"] + q * self.params["tau_v2_minus"]


        v_rhs = ((1 - p) * (1 - v) / tau_v_minus) - ((p * v) / torch.as_tensor(self.params["tau_v_plus"]))
        w_rhs = ((1 - p) * (1 - w) / self.params["tau_w_minus"]) - ((p * w) / self.params["tau_w_plus"])


        # currents
        J_fi = - v * p * (u - self.params["V_c"]) * (1 - u) / self.params["tau_d"]
        J_so = (u * (1 - p) / self.params["tau_0"]) + (p / self.params["tau_r"])
        J_si = - (w * (1 + torch.tanh(self.params["k"] * (u - self.params["V_csi"])))) / (2 * self.params["tau_si"])

        I_ion = -(J_fi + J_so + J_si) / self.params["Cm"]

        # d = torch.sigmoid(d)*0.0009 + 0.0001

        ddu = torch.autograd.grad(du, x, grad_outputs = torch.ones_like(du), retain_graph=True)[0]

        du_xx, du_yy = ddu[:,0], ddu[:,1] 


        eq_a = du_t - torch.as_tensor(d)*(du_xx+du_yy) - I_ion
        eq_b = dv_t - v_rhs
        eq_c = dw_t - w_rhs

        return [eq_a, eq_b, eq_c]
    
    def  heterogeneous_pde_loss(self, x, u, d):
        """
        x = [x,y,t]
        y = [v, w, u, DD]
        """

        v = u[:,0]
        w = u[:,1]
        u = u[:,2]


        du = torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        dw = torch.autograd.grad(w, x, grad_outputs = torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        dv = torch.autograd.grad(v, x, grad_outputs = torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        
        dd = torch.autograd.grad(d, x[:,:2], grad_outputs = torch.ones_like(d), retain_graph = True, create_graph = True)[0]
        dd_x, dd_y = dd[:,0], dd[:,1]

        du_t, du_x, du_y = du[:,2], du[:,0], du[:,1]
        dv_t = dv[:,2]
        dw_t = dw[:,2]

        # gate variables
        p = torch.gt(u, self.params["V_c"]).int()
        q = torch.gt(u, self.params["V_v"]).int()
        tau_v_minus = (1 - q) * self.params["tau_v1_minus"] + q * self.params["tau_v2_minus"]


        v_rhs = ((1 - p) * (1 - v) / tau_v_minus) - ((p * v) / torch.as_tensor(self.params["tau_v_plus"]))
        w_rhs = ((1 - p) * (1 - w) / self.params["tau_w_minus"]) - ((p * w) / self.params["tau_w_plus"])


        # currents
        J_fi = - v * p * (u - self.params["V_c"]) * (1 - u) / self.params["tau_d"]
        J_so = (u * (1 - p) / self.params["tau_0"]) + (p / self.params["tau_r"])
        J_si = - (w * (1 + torch.tanh(self.params["k"] * (u - self.params["V_csi"])))) / (2 * self.params["tau_si"])

        I_ion = -(J_fi + J_so + J_si) / self.params["Cm"]

        # d = torch.sigmoid(d)*0.0009 + 0.0001

        ddu = torch.autograd.grad(du, x, grad_outputs = torch.ones_like(du), retain_graph=True)[0]

        du_xx, du_yy = ddu[:,0], ddu[:,1] 


        eq_a = du_t - d*(du_xx+du_yy) - dd_x*du_x - dd_y*du_y - I_ion
        eq_b = dv_t - v_rhs
        eq_c = dw_t - w_rhs

        return [eq_a, eq_b, eq_c]
        
        
    def BC_func(self, geomtime):
        bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
        return bc
    
    def IC_func(self, observe_train, v_train, w_train, u_train, d_train):

        T_ic = observe_train[:,-1].reshape(-1,1)
        idx_init = np.where(np.isclose(T_ic,1))[0]
        v_init = v_train[idx_init]
        w_init = w_train[idx_init]
        u_init = u_train[idx_init]
        d_init = d_train[idx_init]
        observe_init = observe_train[idx_init]
        return dde.PointSetBC(observe_init,v_init,component=0)
    
    def modify_heter(self, x, y):
        
        x_space, y_space = x[:, 0:1], x[:, 1:2]
        
        x_upper = np.less_equal(x_space, 54*0.1)
        x_lower = np.greater(x_space,32*0.1)
        cond_1 = np.logical_and(x_upper, x_lower)
        
        y_upper = np.less_equal(y_space, 54*0.1)
        y_lower = np.greater(y_space,32*0.1)
        cond_2 = np.logical_and(y_upper, y_lower)
        
        D0 = np.ones_like(x_space)*0.02 
        D1 = np.ones_like(x_space)*0.1
        D = np.where(np.logical_and(cond_1, cond_2),D0,D1)
        return np.concat((y[:,0:2],D), axis=1)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class FNN_D(nn.Module):
    def __init__(self, layer_sizes):
        super(FNN_D, self).__init__()
        self.layer_sizes = layer_sizes
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            nn.init.xavier_uniform_(self.linears[-1].weight)
        
    def forward(self, x):
        for i in range(len(self.layer_sizes)-2):
            x = self.linears[i](x)
            x = torch.tanh(x)
        x = self.linears[-1](x)
        x = torch.sigmoid(x)*0.0009 + 0.0001
        return x


class FNN_AP(nn.Module):
    def __init__(self, layer_sizes):
        super(FNN_AP, self).__init__()
        self.layer_sizes = layer_sizes
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            nn.init.xavier_uniform_(self.linears[-1].weight)
        
    def forward(self, x):
        for i in range(len(self.layer_sizes)-2):
            x = self.linears[i](x)
            x = torch.tanh(x)
        x = self.linears[-1](x)
        x = torch.sigmoid(x)
        return x


# def model_visual_evaluator(model, data):

class model_trainer_u():
    def __init__(self, model_u, model_d, train_data, test_data, epochs, batch_size, geomtime, bc, loss):
        self.model_u = model_u
        self.model_d = model_d
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs
        self.geomtime = geomtime
        self.bc = bc
        
        self.loss = loss

        self.total_test_ap_loss = []
        self.total_test_d_loss = []

        self.total_train_ap_loss = []
        self.total_train_d_loss = []

        self.pinn_test = DataLoader(test_data, batch_size = batch_size, num_workers = 0)

        self.pinn_train = DataLoader(train_data, batch_size = batch_size, num_workers = 0) 

    def train(self):

        self.optimiser_u = torch.optim.Adam(self.model_u.parameters(), 0.0005)
        self.optimiser_d = torch.optim.Adam(self.model_d.parameters(), 0.0005)

        total_ap_loss = 0
        total_d_loss = 0
        
        for i, batch in enumerate(self.pinn_train):
            x_i = batch[:,:3].float()
            x_i.requires_grad = True
            phi_i = batch[:,3].reshape(-1,1).float()
            d_i = batch[:,4].reshape(-1,1).float()

            self.optimiser_u.zero_grad()
            self.optimiser_d.zero_grad()

            # First through the network
            u_i_hat = self.model_u(x_i)
            d_i_hat = self.model_d(x_i[:,:2])

            # Now for the boundary, check which are on the boundary:
            bc_idx = self.geomtime.on_boundary(x_i.detach().cpu().numpy())
            dydx = torch.autograd.grad(u_i_hat, x_i, grad_outputs = torch.ones_like(u_i_hat), retain_graph=True)[0][bc_idx]
            #         dydx = dde.grad.jacobian(u_i_hat, x_i, i = bc.component)[bc_idx]
            # n = bc.boundary_normal(x_i.detach().cpu().numpy(), 0, None)[bc_idx]
            n = torch.as_tensor(self.bc.geom.boundary_normal(x_i.detach().cpu().numpy())[bc_idx])
            bc_loss = torch.sum(abs(dydx*n))

            # Regular loss
            ap_loss = self.loss(u_i_hat, u_i) + bc_loss
            d_loss = self.loss(d_i_hat, d_i)

            ap_loss.backward()
            d_loss.backward()

            self.optimiser_u.step()
            self.optimiser_d.step()
            total_ap_loss += ap_loss.detach().item()
            total_d_loss =+ d_loss.detach().item()
        total_ap_loss /= i+1
        total_d_loss /= i+1
        self.total_train_ap_loss.append(total_ap_loss)
        self.total_train_d_loss.append(total_d_loss)
        print(f'AP loss is {total_ap_loss}, and D loss is {total_d_loss} for epoch 0.')
        print(i)

        test_ap_loss = 0
        test_d_loss = 0
        for i, batch in enumerate(self.pinn_test):
            x_i = batch[:,:3].float()
            x_i.requires_grad = True
            u_i = batch[:,3].reshape(-1,1).float()
            d_i = batch[:,4].reshape(-1,1).float()

            # First through the network
            u_i_hat = self.model_u(x_i)
            d_i_hat = self.model_d(x_i[:,:2])

            # Now for the boundary, check which are on the boundary:
            bc_idx = self.geomtime.on_boundary(x_i.detach().cpu().numpy())
    #         dydx = dde.grad.jacobian(u_i_hat, x_i, i = bc.component)[bc_idx]
            dydx = torch.autograd.grad(u_i_hat, x_i, grad_outputs = torch.ones_like(u_i_hat), retain_graph=True)[0][bc_idx]
            # n = bc.boundary_normal(x_i.detach().cpu().numpy(), 0, None)[bc_idx]
            n = torch.as_tensor(self.bc.geom.boundary_normal(x_i.detach().cpu().numpy())[bc_idx])
            bc_loss = torch.sum(abs(dydx*n))

            # Regular loss
            ap_loss = self.loss(u_i_hat, u_i) + bc_loss
            d_loss = self.loss(d_i_hat, d_i)
            
            test_ap_loss += ap_loss.detach().item()
            test_d_loss =+ d_loss.detach().item()
        test_ap_loss /= i+1
        test_d_loss /= i+1
        self.total_test_ap_loss.append(total_ap_loss)
        self.total_test_d_loss.append(total_d_loss)
        print(f'Test AP loss is {test_ap_loss}, and test D loss is {test_d_loss}')
        print(i)


        self.optimiser_u = torch.optim.Adam(self.model_u.parameters(), 0.00005)
        self.optimiser_d = torch.optim.Adam(self.model_d.parameters(), 0.00005)

        for epoch in range(1,self.epochs):
            total_ap_loss = 0
            total_d_loss = 0
            
            for i, batch in enumerate(self.pinn_train):
                x_i = batch[:,:3].float()
                x_i.requires_grad = True
                u_i = batch[:,3].reshape(-1,1).float()
                d_i = batch[:,4].reshape(-1,1).float()

                self.optimiser_u.zero_grad()
                self.optimiser_d.zero_grad()

                # First through the network
                u_i_hat = self.model_u(x_i)
                d_i_hat = self.model_d(x_i[:,:2])

                # Now for the boundary, check which are on the boundary:
                bc_idx = self.geomtime.on_boundary(x_i.detach().cpu().numpy())
                dydx = torch.autograd.grad(u_i_hat, x_i, grad_outputs = torch.ones_like(u_i_hat), retain_graph=True)[0][bc_idx]
                #         dydx = dde.grad.jacobian(u_i_hat, x_i, i = bc.component)[bc_idx]
                # n = bc.boundary_normal(x_i.detach().cpu().numpy(), 0, None)[bc_idx]
                n = torch.as_tensor(self.bc.geom.boundary_normal(x_i.detach().cpu().numpy())[bc_idx])
                bc_loss = torch.sum(abs(dydx*n))

                # Regular loss
                ap_loss = self.loss(u_i_hat, u_i) + bc_loss
                d_loss = self.loss(d_i_hat, d_i)

                ap_loss.backward()
                d_loss.backward()

                self.optimiser_u.step()
                self.optimiser_d.step()
                total_ap_loss += ap_loss.detach().item()
                total_d_loss =+ d_loss.detach().item()
            total_ap_loss /= i+1
            total_d_loss /= i+1
            self.total_train_ap_loss.append(total_ap_loss)
            self.total_train_d_loss.append(total_d_loss)
            print(f'AP loss is {total_ap_loss}, and D loss is {total_d_loss} for epoch {epoch}.')

            test_ap_loss = 0
            test_d_loss = 0
            for i, batch in enumerate(self.pinn_test):
                x_i = batch[:,:3].float()
                x_i.requires_grad = True
                u_i = batch[:,3].reshape(-1,1).float()
                d_i = batch[:,4].reshape(-1,1).float()

                # First through the network
                u_i_hat = self.model_u(x_i)
                d_i_hat = self.model_d(x_i[:,:2])

                # Now for the boundary, check which are on the boundary:
                bc_idx = self.geomtime.on_boundary(x_i.detach().cpu().numpy())
        #         dydx = dde.grad.jacobian(u_i_hat, x_i, i = bc.component)[bc_idx]
                dydx = torch.autograd.grad(u_i_hat, x_i, grad_outputs = torch.ones_like(u_i_hat), retain_graph=True)[0][bc_idx]
                # n = bc.boundary_normal(x_i.detach().cpu().numpy(), 0, None)[bc_idx]
                n = torch.as_tensor(self.bc.geom.boundary_normal(x_i.detach().cpu().numpy())[bc_idx])
                bc_loss = torch.sum(abs(dydx*n))

                # Regular loss
                ap_loss = self.loss(u_i_hat, u_i) + bc_loss
                d_loss = self.loss(d_i_hat, d_i)
                
                test_ap_loss += ap_loss.detach().item()
                test_d_loss =+ d_loss.detach().item()
            test_ap_loss /= i+1
            test_d_loss /= i+1
            self.total_test_ap_loss.append(total_ap_loss)
            self.total_test_d_loss.append(total_d_loss)
            print(f'Test AP loss is {test_ap_loss}, and test D loss is {test_d_loss}')
        
        
        


class egm_sampler():
    def __init__(self, data, system):
        self.data = data
        self.t = np.arange(np.min(data[:,2]), np.max(data[:,2])+1, system.t_step)
        assert len(self.t)==len(np.unique(self.data[:,2]))
    
    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, idx):
        idx = self.t[idx]
        idx = self.data[:,2]==idx
        sample =  self.data[idx]
        return torch.as_tensor(sample, dtype = torch.float)



class model_trainer_egm_homogeneous():
    def __init__(self, model_u, train_data, test_data, batch_size, geomtime, bc, loss, system):
        self.model_u = model_u
        self.train_data = train_data
        self.test_data = test_data
        self.geomtime = geomtime
        self.bc = bc
        self.t = np.arange(system.min_t, system.max_t, system.t_step)
        self.system = system


        w_x = np.arange( -4, 4+0.01, 0.01) # TODO: maybe here need to add linspace and 256 samples, will see.
        w_y = np.arange( -4, 4+0.01, 0.01)

        w_x = np.linspace( -4, 4+0.01, 257) # TODO: maybe here need to add linspace and 256 samples, will see.
        w_y = np.linspace( -4, 4+0.01, 257)

        w_x, w_y = np.meshgrid(w_x, w_y, sparse = False, indexing = 'ij')
        self.r_egm = 1/np.sqrt(w_x**2 + w_y**2 + 0.01)
        self.r_egm = torch.as_tensor(self.r_egm[np.newaxis,np.newaxis], dtype = torch.float)
        
        self.loss = loss

        self.total_test_ap_loss = []
        self.total_test_d_loss = []

        self.total_train_ap_loss = []
        self.total_train_d_loss = []

        self.batch_size = batch_size

        self.pinn_test = DataLoader(self.test_data, batch_size = batch_size, collate_fn=torch.vstack, num_workers = 0)

        self.pinn_train = DataLoader(self.train_data, batch_size = batch_size, collate_fn=torch.vstack, num_workers = 0) 

        
        

    def train(self, epochs, learning_rate, draw_figure = False, savedir = None, loss_weights = [0,0,0,0], verbose = True):

        self.optimiser_u = torch.optim.Adam(self.model_u.parameters(), learning_rate)
        
        
        losses_log = []
        test_losses_log = []
        
        for epoch in range(epochs):
            total_ap_loss = 0
            losses = [0, 0, 0, 0, 0] # phi, bc, v, w, u
            
            


            for i, batch in enumerate(self.pinn_train):
                self.d = torch.as_tensor(np.tile(self.system.D.reshape(-1,1).squeeze(),len(batch)//(128*128)))
                x_i = batch[:,:3].float()
                x_i.requires_grad = True
                u_i = batch[:,3].reshape(-1,1).reshape((len(batch)//(128*128),1,128,128)).float()

                self.optimiser_u.zero_grad()

                # First through the network
                u_i_hat = self.model_u(x_i)
                dv_y, dv_x = torch.gradient(u_i_hat[...,2].reshape((len(batch)//(128*128), 1, 128,128)), spacing = 0.3125, dim= (2,3))
                dv_yy = torch.gradient(dv_y.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 2)[0]
                dv_xx = torch.gradient(dv_x.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 3)[0]
                ddv = (dv_xx+dv_yy)

                phi_hat = torch.nn.functional.conv2d(  ddv, self.r_egm, padding='same')

                # Now for the boundary, check which are on the boundary:
                bc_idx = self.geomtime.on_boundary(x_i.detach().cpu().numpy())
                dydx = torch.autograd.grad(u_i_hat, x_i, grad_outputs = torch.ones_like(u_i_hat), retain_graph=True)[0][bc_idx]
                n = torch.as_tensor(self.bc.geom.boundary_normal(x_i.detach().cpu().numpy())[bc_idx])
                bc_loss = torch.mean(abs(dydx*n))

                # Let's add residual losses
                res_losses =  self.system.homogeneous_pde_loss(x_i, u_i_hat, self.d)
                u_res = torch.sqrt(torch.mean(torch.pow(res_losses[0],2)))
                v_res = torch.sqrt(torch.mean(torch.pow(res_losses[1],2)))
                w_res = torch.sqrt(torch.mean(torch.pow(res_losses[2],2)))
                
                phi_loss = self.loss(phi_hat, u_i)

                # Regular loss
                l_bc = 1+loss_weights[0]
                l_u = 1+loss_weights[1]
                l_v = 1+loss_weights[2]
                l_w = 1+loss_weights[3]

                ap_loss = phi_loss  + bc_loss*l_bc + u_res*l_u + v_res*l_v + w_res*l_w

                ap_loss.backward()

                self.optimiser_u.step()
                
                # Logging losses
                total_ap_loss += ap_loss.detach().item()
                losses[0]+= phi_loss.detach().item()
                losses[1]+= bc_loss.detach().item()
                losses[2]+=u_res.detach().item()
                losses[3]+=v_res.detach().item()
                losses[4]+=w_res.detach().item()
                
                
                
            total_ap_loss /= i+1
            self.total_train_ap_loss.append(total_ap_loss)
            losses_log.append([a_loss/(i+1) for a_loss in losses])
            
            if epoch%(epochs//100)==0:
                if verbose:
                    print(f'AP loss is {total_ap_loss} for epoch {epoch}, all losses are {losses}')
                if draw_figure==True:
                    fig = plt.figure(figsize = (20,10))
                    ax =plt.subplot(2,4,1)
                    im = ax.imshow(phi_hat[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,5)
                    im = ax.imshow(u_i[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,2)
                    im = ax.imshow(u_i_hat[...,2].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('AP prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,6)
                    im = ax.imshow(batch[:,7].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('AP ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,3)
                    im = ax.imshow(u_i_hat[...,0].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('v prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,7)
                    im = ax.imshow(batch[:,5].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('v ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,4)
                    im = ax.imshow(u_i_hat[...,1].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('w prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,8)
                    im = ax.imshow(batch[:,6].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('w ground truth')
                    plt.colorbar(im, ax = ax)

                    if savedir is not None:
                        plt.savefig(os.path.join( savedir, f'train_{epoch}.png'), bbox_inches='tight')
                        torch.save(self.model_u, os.path.join(savedir, f'model_u_{epoch}.pt'))


                    plt.show()
                    if verbose:
                        print(50*'++')
                    
                    
            losses = [0, 0, 0, 0, 0]
            test_ap_loss = 0
            for i, batch in enumerate(self.pinn_test):
                self.d = torch.as_tensor(np.tile(self.system.D.reshape(-1,1).squeeze(),len(batch)//(128*128)))
                x_i = batch[:,:3].float()
                x_i.requires_grad = True
                u_i = batch[:,3].reshape(-1,1).reshape((len(batch)//(128*128),1,128,128)).float()

                # First through the network
                u_i_hat = self.model_u(x_i)
                dv_y, dv_x = torch.gradient(u_i_hat[...,2].reshape((len(batch)//(128*128), 1, 128,128)), spacing = 0.3125, dim= (2,3))
                dv_yy = torch.gradient(dv_y.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 2)[0]
                dv_xx = torch.gradient(dv_x.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 3)[0]
                ddv = (dv_xx+dv_yy)

                phi_hat = torch.nn.functional.conv2d(  ddv, self.r_egm, padding='same')

                # Now for the boundary, check which are on the boundary:
                bc_idx = self.geomtime.on_boundary(x_i.detach().cpu().numpy())
                dydx = torch.autograd.grad(u_i_hat, x_i, grad_outputs = torch.ones_like(u_i_hat), retain_graph=True)[0][bc_idx]
                n = torch.as_tensor(self.bc.geom.boundary_normal(x_i.detach().cpu().numpy())[bc_idx])
                bc_loss = torch.mean(abs(dydx*n))

                # Let's add residual losses
                res_losses =  self.system.homogeneous_pde_loss(x_i, u_i_hat, self.d)
                u_res = torch.sqrt(torch.mean(torch.pow(res_losses[0],2)))
                v_res = torch.sqrt(torch.mean(torch.pow(res_losses[1],2)))
                w_res = torch.sqrt(torch.mean(torch.pow(res_losses[2],2)))

                phi_loss = self.loss(phi_hat, u_i)

                # Regular loss
                ap_loss = phi_loss + bc_loss + u_res + v_res + w_res

                test_ap_loss += ap_loss.detach().item()
                losses[0]+= phi_loss.detach().item()
                losses[1]+= bc_loss.detach().item()
                losses[2]+=u_res.detach().item()
                losses[3]+=v_res.detach().item()
                losses[4]+=w_res.detach().item()
                break

            test_ap_loss /= i+1
            test_losses_log.append([a_loss/(i+1) for a_loss in losses])

            self.total_test_ap_loss.append(test_ap_loss)
            
            if epoch%(epochs//100)==0:

                if verbose:
                    print(f'AP loss is {test_ap_loss} for epoch {epoch}, all losses are {losses}')
                if draw_figure==True:
                    fig = plt.figure(figsize = (20,10))
                    ax =plt.subplot(2,4,1)
                    im = ax.imshow(phi_hat[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,5)
                    im = ax.imshow(u_i[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,2)
                    im = ax.imshow(u_i_hat[...,2].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('AP prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,6)
                    im = ax.imshow(batch[:,7].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('AP ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,3)
                    im = ax.imshow(u_i_hat[...,0].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('v prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,7)
                    im = ax.imshow(batch[:,5].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('v ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,4)
                    im = ax.imshow(u_i_hat[...,1].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('w prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,8)
                    im = ax.imshow(batch[:,6].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('w ground truth')
                    plt.colorbar(im, ax = ax)

                    if savedir is not None:
                        plt.savefig(os.path.join( savedir, f'test_epoch{epoch}.png'), bbox_inches='tight')

                    plt.show()
                    if verbose:
                        print(50*'++')
                        print(50*'++')
                        print(50*'++')

        fig = plt.figure(figsize = (20,10))
        ax =plt.subplot(2,4,1)
        im = ax.imshow(phi_hat[-1][0].detach().cpu().numpy())
        ax.set_title('EGM prediction')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,5)
        im = ax.imshow(u_i[-1][0].detach().cpu().numpy())
        ax.set_title('EGM ground truth')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,2)
        im = ax.imshow(u_i_hat[...,2].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
        ax.set_title('AP prediction')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,6)
        im = ax.imshow(batch[:,7].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
        ax.set_title('AP ground truth')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,3)
        im = ax.imshow(u_i_hat[...,0].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
        ax.set_title('v prediction')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,7)
        im = ax.imshow(batch[:,5].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
        ax.set_title('v ground truth')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,4)
        im = ax.imshow(u_i_hat[...,1].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
        ax.set_title('w prediction')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,8)
        im = ax.imshow(batch[:,6].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
        ax.set_title('w ground truth')
        plt.colorbar(im, ax = ax)

        if savedir is not None:
            plt.savefig(os.path.join( savedir, f'train_{epoch}.png'), bbox_inches='tight')
            torch.save(self.model_u, os.path.join(savedir, f'model_u_{epoch}.pt'))
        return self.model_u, [losses_log, test_losses_log]

    def incremental_train(self, epochs, learning_rate, draw_figure = False, savedir = None, loss_weights = [0,0,0,0]):

        increment = [0]
        while len(increment) <= len(self.train_data):

            self.pinn_train = DataLoader(Subset(self.train_data, increment), batch_size = self.batch_size, collate_fn=torch.vstack, num_workers = 0) 
            

            self.pinn_test = DataLoader(Subset(self.train_data, [increment[-1]+1]), batch_size = self.batch_size, collate_fn=torch.vstack, num_workers = 0)

            self.model_u, losses = self.train( epochs//len(self.train_data), learning_rate, draw_figure, savedir, loss_weights )
            increment.append(increment[-1]+1)
        return self.model_u, losses


class model_trainer_phi_homogeneous():
    def __init__(self, model_u, train_data, test_data, batch_size, geomtime, bc, loss, system):
        self.model_u = model_u
        self.train_data = train_data
        self.test_data = test_data
        self.geomtime = geomtime
        self.bc = bc
        self.t = np.arange(system.min_t, system.max_t, system.t_step)
        self.system = system


        w_x = np.arange( -4, 4+0.01, 0.01) # TODO: maybe here need to add linspace and 256 samples, will see.
        w_y = np.arange( -4, 4+0.01, 0.01)

        w_x = np.linspace( -4, 4+0.01, 257) # TODO: maybe here need to add linspace and 256 samples, will see.
        w_y = np.linspace( -4, 4+0.01, 257)

        w_x, w_y = np.meshgrid(w_x, w_y, sparse = False, indexing = 'ij')
        self.r_egm = 1/np.sqrt(w_x**2 + w_y**2 + 0.01)
        self.r_egm = torch.as_tensor(self.r_egm[np.newaxis,np.newaxis], dtype = torch.float)
        
        self.loss = loss

        self.total_test_ap_loss = []
        self.total_test_d_loss = []

        self.total_train_ap_loss = []
        self.total_train_d_loss = []

        self.batch_size = batch_size

        self.pinn_test = DataLoader(self.test_data, batch_size = batch_size, collate_fn=torch.vstack, num_workers = 0)

        self.pinn_train = DataLoader(self.train_data, batch_size = batch_size, collate_fn=torch.vstack, num_workers = 0) 

        
        

    def train(self, epochs, learning_rate, draw_figure = False, savedir = None, loss_weights = [0,0,0,0], verbose = True):

        self.optimiser_u = torch.optim.Adam(self.model_u.parameters(), learning_rate)
        
        
        losses_log = []
        test_losses_log = []
        
        for epoch in range(epochs):
            total_ap_loss = 0
            losses = [0, 0, 0, 0, 0] # phi, bc, v, w, u
            
            


            for i, batch in enumerate(self.pinn_train):
                self.d = torch.as_tensor(np.tile(self.system.D.reshape(-1,1).squeeze(),len(batch)//(128*128)))
                x_i = batch[:,:4].float()
                x_i.requires_grad = True
                u_i = 20*batch[:,3].reshape(-1,1).reshape((len(batch)//(128*128),1,128,128)).float()

                self.optimiser_u.zero_grad()

                # First through the network
                u_i_hat = self.model_u(x_i)
                dv_y, dv_x = torch.gradient(u_i_hat[...,2].reshape((len(batch)//(128*128), 1, 128,128)), spacing = 0.3125, dim= (2,3))
                dv_yy = torch.gradient(dv_y.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 2)[0]
                dv_xx = torch.gradient(dv_x.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 3)[0]
                ddv = (dv_xx+dv_yy)

                phi_hat = torch.nn.functional.conv2d(  ddv, self.r_egm, padding='same')

                # Now for the boundary, check which are on the boundary:
                bc_idx = self.geomtime.on_boundary(x_i[:,:3].detach().cpu().numpy())
                dydx = torch.autograd.grad(u_i_hat, x_i, grad_outputs = torch.ones_like(u_i_hat), retain_graph=True)[0][bc_idx,:3]
                n = torch.as_tensor(self.bc.geom.boundary_normal(x_i[:,:3].detach().cpu().numpy())[bc_idx])
                bc_loss = torch.mean(abs(dydx*n))

                # Let's add residual losses
                res_losses =  self.system.homogeneous_pde_loss(x_i, u_i_hat, self.d)
                u_res = torch.sqrt(torch.mean(torch.pow(res_losses[0],2)))
                v_res = torch.sqrt(torch.mean(torch.pow(res_losses[1],2)))
                w_res = torch.sqrt(torch.mean(torch.pow(res_losses[2],2)))
                
                phi_loss = self.loss(phi_hat, u_i)

                # Regular loss
                l_bc = 1+loss_weights[0]
                l_u = 1+loss_weights[1]
                l_v = 1+loss_weights[2]
                l_w = 1+loss_weights[3]

                ap_loss = phi_loss  + bc_loss*l_bc + u_res*l_u + v_res*l_v + w_res*l_w

                ap_loss.backward()

                self.optimiser_u.step()
                
                # Logging losses
                total_ap_loss += ap_loss.detach().item()
                losses[0]+= phi_loss.detach().item()
                losses[1]+= bc_loss.detach().item()
                losses[2]+=u_res.detach().item()
                losses[3]+=v_res.detach().item()
                losses[4]+=w_res.detach().item()
                
                
                
            total_ap_loss /= i+1
            self.total_train_ap_loss.append(total_ap_loss)
            losses_log.append([a_loss/(i+1) for a_loss in losses])
            
            if epoch%(epochs//100)==0:
                if verbose:
                    print(f'AP loss is {total_ap_loss} for epoch {epoch}, all losses are {losses}')
                if draw_figure==True:
                    fig = plt.figure(figsize = (20,10))
                    ax =plt.subplot(2,4,1)
                    im = ax.imshow(phi_hat[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,5)
                    im = ax.imshow(u_i[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,2)
                    im = ax.imshow(u_i_hat[...,2].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('AP prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,6)
                    im = ax.imshow(batch[:,7].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('AP ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,3)
                    im = ax.imshow(u_i_hat[...,0].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('v prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,7)
                    im = ax.imshow(batch[:,5].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('v ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,4)
                    im = ax.imshow(u_i_hat[...,1].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('w prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,8)
                    im = ax.imshow(batch[:,6].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('w ground truth')
                    plt.colorbar(im, ax = ax)

                    if savedir is not None:
                        plt.savefig(os.path.join( savedir, f'train_{epoch}.png'), bbox_inches='tight')
                        torch.save(self.model_u, os.path.join(savedir, f'model_u_{epoch}.pt'))


                    plt.show()
                    if verbose:
                        print(50*'++')
                    
                    
            losses = [0, 0, 0, 0, 0]
            test_ap_loss = 0
            for i, batch in enumerate(self.pinn_test):
                self.d = torch.as_tensor(np.tile(self.system.D.reshape(-1,1).squeeze(),len(batch)//(128*128)))
                x_i = batch[:,:4].float()
                x_i.requires_grad = True
                u_i = 20*batch[:,3].reshape(-1,1).reshape((len(batch)//(128*128),1,128,128)).float()

                # First through the network
                u_i_hat = self.model_u(x_i)
                dv_y, dv_x = torch.gradient(u_i_hat[...,2].reshape((len(batch)//(128*128), 1, 128,128)), spacing = 0.3125, dim= (2,3))
                dv_yy = torch.gradient(dv_y.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 2)[0]
                dv_xx = torch.gradient(dv_x.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 3)[0]
                ddv = (dv_xx+dv_yy)

                phi_hat = torch.nn.functional.conv2d(  ddv, self.r_egm, padding='same')

                # Now for the boundary, check which are on the boundary:
                bc_idx = self.geomtime.on_boundary(x_i[:,:3].detach().cpu().numpy())
                dydx = torch.autograd.grad(u_i_hat, x_i, grad_outputs = torch.ones_like(u_i_hat), retain_graph=True)[0][bc_idx,:3]
                n = torch.as_tensor(self.bc.geom.boundary_normal(x_i[:,:3].detach().cpu().numpy())[bc_idx])
                bc_loss = torch.mean(abs(dydx*n))

                # Let's add residual losses
                res_losses =  self.system.homogeneous_pde_loss(x_i, u_i_hat, self.d)
                u_res = torch.sqrt(torch.mean(torch.pow(res_losses[0],2)))
                v_res = torch.sqrt(torch.mean(torch.pow(res_losses[1],2)))
                w_res = torch.sqrt(torch.mean(torch.pow(res_losses[2],2)))
                
                phi_loss = self.loss(phi_hat, u_i)

                # Regular loss
                ap_loss = phi_loss + bc_loss + u_res + v_res + w_res

                test_ap_loss += ap_loss.detach().item()
                losses[0]+= phi_loss.detach().item()
                losses[1]+= bc_loss.detach().item()
                losses[2]+=u_res.detach().item()
                losses[3]+=v_res.detach().item()
                losses[4]+=w_res.detach().item()
                break

            test_ap_loss /= i+1
            test_losses_log.append([a_loss/(i+1) for a_loss in losses])

            self.total_test_ap_loss.append(test_ap_loss)
            
            if epoch%(epochs//100)==0:

                if verbose:
                    print(f'AP loss is {test_ap_loss} for epoch {epoch}, all losses are {losses}')
                if draw_figure==True:
                    fig = plt.figure(figsize = (20,10))
                    ax =plt.subplot(2,4,1)
                    im = ax.imshow(phi_hat[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,5)
                    im = ax.imshow(u_i[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,2)
                    im = ax.imshow(u_i_hat[...,2].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('AP prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,6)
                    im = ax.imshow(batch[:,7].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('AP ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,3)
                    im = ax.imshow(u_i_hat[...,0].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('v prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,7)
                    im = ax.imshow(batch[:,5].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('v ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,4)
                    im = ax.imshow(u_i_hat[...,1].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('w prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,8)
                    im = ax.imshow(batch[:,6].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('w ground truth')
                    plt.colorbar(im, ax = ax)

                    if savedir is not None:
                        plt.savefig(os.path.join( savedir, f'test_epoch{epoch}.png'), bbox_inches='tight')

                    plt.show()
                    if verbose:
                        print(50*'++')
                        print(50*'++')
                        print(50*'++')

        fig = plt.figure(figsize = (20,10))
        ax =plt.subplot(2,4,1)
        im = ax.imshow(phi_hat[-1][0].detach().cpu().numpy())
        ax.set_title('EGM prediction')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,5)
        im = ax.imshow(u_i[-1][0].detach().cpu().numpy())
        ax.set_title('EGM ground truth')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,2)
        im = ax.imshow(u_i_hat[...,2].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
        ax.set_title('AP prediction')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,6)
        im = ax.imshow(batch[:,7].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
        ax.set_title('AP ground truth')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,3)
        im = ax.imshow(u_i_hat[...,0].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
        ax.set_title('v prediction')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,7)
        im = ax.imshow(batch[:,5].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
        ax.set_title('v ground truth')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,4)
        im = ax.imshow(u_i_hat[...,1].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
        ax.set_title('w prediction')
        plt.colorbar(im, ax = ax)
        
        ax =plt.subplot(2,4,8)
        im = ax.imshow(batch[:,6].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
        ax.set_title('w ground truth')
        plt.colorbar(im, ax = ax)

        if savedir is not None:
            plt.savefig(os.path.join( savedir, f'train_{epoch}.png'), bbox_inches='tight')
            torch.save(self.model_u, os.path.join(savedir, f'model_u_{epoch}.pt'))
        return self.model_u, [losses_log, test_losses_log]

    def incremental_train(self, epochs, learning_rate, draw_figure = False, savedir = None, loss_weights = [0,0,0,0]):

        increment = [0]
        while len(increment) <= len(self.train_data):

            self.pinn_train = DataLoader(Subset(self.train_data, increment), batch_size = self.batch_size, collate_fn=torch.vstack, num_workers = 0) 
            

            self.pinn_test = DataLoader(Subset(self.train_data, [increment[-1]+1]), batch_size = self.batch_size, collate_fn=torch.vstack, num_workers = 0)

            self.model_u, losses = self.train( epochs//len(self.train_data), learning_rate, draw_figure, savedir, loss_weights )
            increment.append(increment[-1]+1)
        return self.model_u, losses
    
class model_trainer_egm_homogeneous_no_pinn():
    def __init__(self, model_u, train_data, test_data, batch_size, geomtime, bc, loss, system):
        self.model_u = model_u
        self.train_data = train_data
        self.test_data = test_data
        self.geomtime = geomtime
        self.bc = bc
        self.t = np.arange(system.min_t, system.max_t, system.t_step)
        self.system = system

        w_x = np.arange( -4, 4+0.01, 0.01) # TODO: maybe here need to add linspace and 256 samples, will see.
        w_y = np.arange( -4, 4+0.01, 0.01)

        w_x = np.linspace( -4, 4+0.01, 257) # TODO: maybe here need to add linspace and 256 samples, will see.
        w_y = np.linspace( -4, 4+0.01, 257)

        w_x, w_y = np.meshgrid(w_x, w_y, sparse = False, indexing = 'ij')
        self.r_egm = 1/np.sqrt(w_x**2 + w_y**2 + 0.01)
        self.r_egm = torch.as_tensor(self.r_egm[np.newaxis,np.newaxis], dtype = torch.float)
        
        self.loss = loss

        self.total_test_ap_loss = []
        self.total_test_d_loss = []

        self.total_train_ap_loss = []
        self.total_train_d_loss = []

        self.batch_size = batch_size

        self.pinn_test = DataLoader(self.test_data, batch_size = batch_size, collate_fn=torch.vstack, num_workers = 0)

        self.pinn_train = DataLoader(self.train_data, batch_size = batch_size, collate_fn=torch.vstack, num_workers = 0) 
        
        

    def train(self, epochs, learning_rate, draw_figure = False, savedir = None, loss_weights = [0,0,0,0]):

        self.optimiser_u = torch.optim.Adam(self.model_u.parameters(), learning_rate)
        
        
        losses_log = []
        test_losses_log = []
        
        for epoch in range(epochs):
            total_ap_loss = 0
            losses = [0, 0, 0, 0, 0] # phi, bc, v, w, u
            
            


            for i, batch in enumerate(self.pinn_train):
                self.d = torch.as_tensor(np.tile(self.system.D.reshape(-1,1).squeeze(),len(batch)//(128*128)))
                x_i = batch[:,:3].float()
                x_i.requires_grad = True
                u_i = batch[:,3].reshape(-1,1).reshape((len(batch)//(128*128),1,128,128)).float()

                self.optimiser_u.zero_grad()

                # First through the network
                u_i_hat = self.model_u(x_i)
                dv_y, dv_x = torch.gradient(u_i_hat[...,2].reshape((len(batch)//(128*128), 1, 128,128)), spacing = 0.3125, dim= (2,3))
                dv_yy = torch.gradient(dv_y.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 2)[0]
                dv_xx = torch.gradient(dv_x.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 3)[0]
                ddv = (dv_xx+dv_yy)

                phi_hat = torch.nn.functional.conv2d(  ddv, self.r_egm, padding='same')

                

                ap_loss = self.loss(phi_hat, u_i)

                ap_loss.backward()

                self.optimiser_u.step()
                
                # Logging losses
                total_ap_loss += ap_loss.detach().item()
                
                
            total_ap_loss /= i+1
            self.total_train_ap_loss.append(total_ap_loss)
            
            if epoch%(epochs//100)==0:

                print(f'AP loss is {total_ap_loss} for epoch {epoch}, all losses are {losses}')
                if draw_figure==True:
                    fig = plt.figure(figsize = (20,10))
                    ax =plt.subplot(2,4,1)
                    im = ax.imshow(phi_hat[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,5)
                    im = ax.imshow(u_i[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,2)
                    im = ax.imshow(u_i_hat[...,2].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('AP prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,6)
                    im = ax.imshow(batch[:,7].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('AP ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,3)
                    im = ax.imshow(u_i_hat[...,0].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('v prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,7)
                    im = ax.imshow(batch[:,5].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('v ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,4)
                    im = ax.imshow(u_i_hat[...,1].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('w prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,8)
                    im = ax.imshow(batch[:,6].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('w ground truth')
                    plt.colorbar(im, ax = ax)

                    if savedir is not None:
                        plt.savefig(os.path.join( savedir, f'train_{epoch}.png'), bbox_inches='tight')
                        torch.save(self.model_u, os.path.join(savedir, f'model_u_{epoch}.pt'))


                    plt.show()
                    print(50*'++')
                    
                    

            losses = [0, 0, 0, 0, 0]
            test_ap_loss = 0
            for i, batch in enumerate(self.pinn_test):
                self.d = torch.as_tensor(np.tile(self.system.D.reshape(-1,1).squeeze(),len(batch)//(128*128)))
                x_i = batch[:,:3].float()
                x_i.requires_grad = True
                u_i = batch[:,3].reshape(-1,1).reshape((len(batch)//(128*128),1,128,128)).float()

                # First through the network
                u_i_hat = self.model_u(x_i)
                dv_y, dv_x = torch.gradient(u_i_hat[...,2].reshape((len(batch)//(128*128), 1, 128,128)), spacing = 0.3125, dim= (2,3))
                dv_yy = torch.gradient(dv_y.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 2)[0]
                dv_xx = torch.gradient(dv_x.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 3)[0]
                ddv = (dv_xx+dv_yy)

                phi_hat = torch.nn.functional.conv2d(  ddv, self.r_egm, padding='same')

                

                # Regular loss
                ap_loss = self.loss(phi_hat, u_i)

                test_ap_loss += ap_loss.detach().item()
                break

            test_ap_loss /= i+1

            self.total_test_ap_loss.append(test_ap_loss)
            
            if epoch%(epochs//100)==0:

                print(f'AP loss is {test_ap_loss} for epoch {epoch}, all losses are {losses}')
                if draw_figure==True:
                    fig = plt.figure(figsize = (20,10))
                    ax =plt.subplot(2,4,1)
                    im = ax.imshow(phi_hat[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,5)
                    im = ax.imshow(u_i[-1][0].detach().cpu().numpy())
                    ax.set_title('EGM ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,2)
                    im = ax.imshow(u_i_hat[...,2].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('AP prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,6)
                    im = ax.imshow(batch[:,7].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('AP ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,3)
                    im = ax.imshow(u_i_hat[...,0].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('v prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,7)
                    im = ax.imshow(batch[:,5].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('v ground truth')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,4)
                    im = ax.imshow(u_i_hat[...,1].reshape((len(batch)//(128*128), 128, 128))[-1].detach().cpu().numpy())
                    ax.set_title('w prediction')
                    plt.colorbar(im, ax = ax)
                    
                    ax =plt.subplot(2,4,8)
                    im = ax.imshow(batch[:,6].reshape(-1,1).reshape((len(batch)//(128*128),128,128))[-1].detach().cpu().numpy())
                    ax.set_title('w ground truth')
                    plt.colorbar(im, ax = ax)

                    if savedir is not None:
                        plt.savefig(os.path.join( savedir, f'test_epoch{epoch}.png'), bbox_inches='tight')

                    plt.show()
                    print(50*'++')
                    print(50*'++')
                    print(50*'++')
        # print(i)
        return self.model_u, [losses_log, test_losses_log]



class model_trainer_egm_heterogeneous():
    def __init__(self, model_u, model_d, train_data, test_data, epochs, batch_size, geomtime, bc, loss, system):
        self.model_u = model_u
        self.model_d = model_d
        self.train_data = egm_sampler(train_data, system)
        self.test_data = egm_sampler(test_data, system)
        self.epochs = epochs
        self.geomtime = geomtime
        self.bc = bc
        self.t = np.arange(system.min_t, system.max_t, system.t_step)
        self.system = system

        w_x = np.arange( -4, 4+0.01, 0.01) # TODO: maybe here need to add linspace and 256 samples, will see.
        w_y = np.arange( -4, 4+0.01, 0.01)

        w_x = np.linspace( -4, 4+0.01, 129) # TODO: maybe here need to add linspace and 256 samples, will see.
        w_y = np.linspace( -4, 4+0.01, 129)

        w_x, w_y = np.meshgrid(w_x, w_y, sparse = False, indexing = 'ij')
        self.r_egm = 1/np.sqrt(w_x**2 + w_y**2 + 0.01)
        self.r_egm = torch.as_tensor(self.r_egm[np.newaxis,np.newaxis], dtype = torch.float)
        
        self.loss = loss

        self.total_test_ap_loss = []
        self.total_test_d_loss = []

        self.total_train_ap_loss = []
        self.total_train_d_loss = []

        self.batch_size = batch_size

        self.pinn_test = DataLoader(self.test_data, batch_size = batch_size, collate_fn=torch.vstack, num_workers = 0)

        self.pinn_train = DataLoader(self.train_data, batch_size = batch_size, collate_fn=torch.vstack, num_workers = 0) 
        
        self.d = torch.as_tensor(np.tile(system.D.reshape(-1,1).squeeze(),self.batch_size))

    def train(self, draw_figure = False):

        self.optimiser_u = torch.optim.Adam(self.model_u.parameters(), 0.0001)
        self.optimiser_d = torch.optim.Adam(self.model_d.parameters(), 0.0001)
        
        for epoch in range(self.epochs):
            total_ap_loss = 0
            total_d_loss = 0


            for i, batch in enumerate(self.pinn_train):
                self.d = torch.as_tensor(np.tile(self.system.D.reshape(-1,1).squeeze(),len(batch)//(128*128)))
                x_i = batch[:,:3].float()
                x_i.requires_grad = True
                u_i = batch[:,3].reshape(-1,1).reshape((len(batch)//(128*128),1,128,128)).float()

                self.optimiser_u.zero_grad()
                self.optimiser_d.zero_grad()

                # First through the network
                u_i_hat = self.model_u(x_i)
                d_i_hat = self.model_d(x_i[:,:2]).reshape((len(batch)//(128*128), 1, 128, 128))
                dv_y, dv_x = torch.gradient(u_i_hat[...,2].reshape((len(batch)//(128*128), 1, 128,128)), spacing = 0.3125, dim= (2,3))
                dv_yy = torch.gradient(d_i_hat*dv_y.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 2)[0]
                dv_xx = torch.gradient(d_i_hat*dv_x.reshape((len(batch)//(128*128), 1,128,128)), spacing = 0.3125, dim= 3)[0]
                ddv = (dv_xx+dv_yy)

                phi_hat = torch.nn.functional.conv2d(  ddv, self.r_egm, padding='same')

                # Now for the boundary, check which are on the boundary:
                bc_idx = self.geomtime.on_boundary(x_i.detach().cpu().numpy())
                dydx = torch.autograd.grad(u_i_hat, x_i, grad_outputs = torch.ones_like(u_i_hat), retain_graph=True)[0][bc_idx]
                n = torch.as_tensor(self.bc.geom.boundary_normal(x_i.detach().cpu().numpy())[bc_idx])
                bc_loss = torch.sum(abs(dydx*n))

                # Let's add residual losses
                res_losses =  self.system.heterogeneous_pde_loss(x_i, u_i_hat, d_i_hat)
                u_res = torch.sqrt(torch.mean(torch.pow(res_losses[0],2)))
                v_res = torch.sqrt(torch.mean(torch.pow(res_losses[1],2)))
                w_res = torch.sqrt(torch.mean(torch.pow(res_losses[2],2)))

                # Regular loss
                ap_loss = self.loss(phi_hat, u_i) + bc_loss + u_res + v_res + w_res

                ap_loss.backward()
                

                self.optimiser_u.step()
                self.optimiser_d.step()
                
                total_ap_loss += ap_loss.detach().item()
            total_ap_loss /= i+1
            self.total_train_ap_loss.append(total_ap_loss)
            
            if epoch%(self.epochs//10)==0:

                print(f'AP loss is {total_ap_loss} for epoch {epoch}.')
                if draw_figure==True:
                    fig = plt.figure(figsize = (10,5))
                    ax =plt.subplot(1,3,1)
                    im = ax.imshow(phi_hat[0][0].detach().cpu().numpy())
                    plt.colorbar(im, ax = ax)
                    ax =plt.subplot(1,3,2)
                    im = ax.imshow(u_i[0][0].detach().cpu().numpy())
                    plt.colorbar(im, ax = ax)
                    ax =plt.subplot(1,3,3)
                    im = ax.imshow(u_i_hat[...,2].reshape((len(batch)//(128*128), 128, 128))[0].detach().cpu().numpy())
                    plt.colorbar(im, ax = ax)
                    plt.show()


            test_ap_loss = 0
            for i, batch in enumerate(self.pinn_test):
                self.d = torch.as_tensor(np.tile(self.system.D.reshape(-1,1).squeeze(),len(batch)//(128*128)))
                x_i = batch[:,:3].float()
                x_i.requires_grad = True
                u_i = batch[:,3].reshape(-1,1).reshape((len(batch)//(128*128),1,128,128)).float()

                # First through the network
                u_i_hat = self.model_u(x_i)
                dv_y, dv_x = torch.gradient(u_i_hat[...,2].reshape((len(batch)//(128*128), 1, 128,128)), spacing = 0.3125, dim= (2,3))
                dv_yy = torch.gradient(dv_y.reshape((self.batch_size, 1,128,128)), spacing = 0.3125, dim= 2)[0]
                dv_xx = torch.gradient(dv_x.reshape((self.batch_size, 1,128,128)), spacing = 0.3125, dim= 3)[0]
                ddv = (dv_xx+dv_yy)

                phi_hat = torch.nn.functional.conv2d(  ddv, self.r_egm, padding='same')

                # Now for the boundary, check which are on the boundary:
                bc_idx = self.geomtime.on_boundary(x_i.detach().cpu().numpy())
                dydx = torch.autograd.grad(u_i_hat, x_i, grad_outputs = torch.ones_like(u_i_hat), retain_graph=True)[0][bc_idx]
                n = torch.as_tensor(self.bc.geom.boundary_normal(x_i.detach().cpu().numpy())[bc_idx])
                bc_loss = torch.sum(abs(dydx*n))

                # Let's add residual losses
                res_losses =  self.system.homogeneous_pde_loss(x_i, u_i_hat, d_i_hat)
                u_res = torch.sqrt(torch.mean(torch.pow(res_losses[0],2)))
                v_res = torch.sqrt(torch.mean(torch.pow(res_losses[1],2)))
                w_res = torch.sqrt(torch.mean(torch.pow(res_losses[2],2)))

                # Regular loss
                ap_loss = self.loss(phi_hat, u_i) + bc_loss + u_res + v_res + w_res

                test_ap_loss += ap_loss.detach().item()
            test_ap_loss /= i+1
            self.total_test_ap_loss.append(total_ap_loss)
            
            if epoch%(self.epochs//10)==0:

                print(f'Test AP loss is {test_ap_loss}')
                if draw_figure==True:
                    fig = plt.figure(figsize = (10,5))
                    ax =plt.subplot(1,3,1)
                    im = ax.imshow(phi_hat[0][0].detach().cpu().numpy())
                    plt.colorbar(im, ax = ax)
                    ax =plt.subplot(1,3,2)
                    im = ax.imshow(u_i[0][0].detach().cpu().numpy())
                    plt.colorbar(im, ax = ax)
                    ax =plt.subplot(1,3,3)
                    im = ax.imshow(u_i_hat[...,2].reshape((self.batch_size,128, 128))[0].detach().cpu().numpy())
                    plt.colorbar(im, ax = ax)
                    plt.show()
        # print(i)
        return self.model_u