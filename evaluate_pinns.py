from egm_pinns import *
import torch 
import os
import argparse
import pickle
from sklearn.model_selection import train_test_split
import matplotlib
import IPython
from matplotlib.animation import FuncAnimation, PillowWriter

from IPython.display import HTML


def read_case(case_name):
    case_dir = {}
    case = case_name.split('_')
    if case[0] == 'forward':
        case_dir['forward']=True
        case_dir['filename'] = '_'.join(case[2:6]) +'.hdf5'
        case_dir['start_time'] = int(case[6])
        case_dir['end_time'] = int(case[7])
        case_dir['step'] = int(case[8])
        case_dir['hidden_layer_size'] = int(case[9])
        case_dir['learning_rate'] = float(case[10])
        case_dir['split'] = 0.1*float(case[11])
        case_dir['epochs'] = int(case[12])
        case_dir['l_bc'] = float(case[13])
        case_dir['l_u'] = float(case[14])
        case_dir['l_v'] = float(case[15])
        case_dir['l_w'] = float(case[16])
    else:
        case_dir['forward']=False
        case_dir['filename'] = '_'.join(case[1:5]) +'.hdf5'
        case_dir['start_time'] = int(case[5])
        case_dir['end_time'] = int(case[6])
        case_dir['step'] = int(case[7])
        case_dir['hidden_layer_size'] = int(case[8])
        case_dir['learning_rate'] = float(case[9])
        case_dir['split'] = 0.1*float(case[10])
        case_dir['epochs'] = int(case[11])
        case_dir['l_bc'] = float(case[12])
        case_dir['l_u'] = float(case[13])
        case_dir['l_v'] = float(case[14])
        case_dir['l_w'] = float(case[15])
    
    return case_dir

def get_predicted_state(sample, trainer):
    u_i_hat = trainer.model_u(sample[:,:3])
    dv_y, dv_x = torch.gradient(u_i_hat[...,2].reshape((1, 1, 128,128)), spacing = 0.3125, dim= (2,3))
    dv_yy = torch.gradient(dv_y.reshape((1, 1,128,128)), spacing = 0.3125, dim= 2)[0]
    dv_xx = torch.gradient(dv_x.reshape((1, 1,128,128)), spacing = 0.3125, dim= 3)[0]
    ddv = (dv_xx+dv_yy)

    phi_hat = torch.nn.functional.conv2d(  ddv, trainer.r_egm, padding='same').squeeze().detach().cpu().numpy()
    v_hat = u_i_hat[...,0].reshape((128,128)).detach().cpu().numpy()
    u_hat = u_i_hat[...,2].reshape((128,128)).detach().cpu().numpy()
    w_hat = u_i_hat[...,1].reshape((128,128)).detach().cpu().numpy()
    
    return [phi_hat, u_hat, v_hat, w_hat]

def get_gt_state(sample, trainer):
    
#     dv_y, dv_x = torch.gradient(sample[:,3].reshape((1, 1, 128,128)), spacing = 0.3125, dim= (2,3))
#     dv_yy = torch.gradient(dv_y.reshape((1, 1,128,128)), spacing = 0.3125, dim= 2)[0]
#     dv_xx = torch.gradient(dv_x.reshape((1, 1,128,128)), spacing = 0.3125, dim= 3)[0]
#     ddv = (dv_xx+dv_yy)

#     phi = torch.nn.functional.conv2d(  ddv, trainer.r_egm, padding='same').squeeze().detach().cpu().numpy()
    phi = sample[...,3].reshape((128,128)).detach().cpu().numpy()
    v = sample[...,5].reshape((128,128)).detach().cpu().numpy()
    u = sample[...,7].reshape((128,128)).detach().cpu().numpy()
    w= sample[...,6].reshape((128,128)).detach().cpu().numpy()
    
    return [phi, u, v, w]

def animate(root, case_name, filename, start_time_test = None, stop_time_test = None):
    case_dir = read_case(case_name)
    
    simulation_name = case_dir['filename']
    start_time = case_dir['start_time']
    stop_time = case_dir['end_time']
    step_dt = case_dir['step']
    hidden_layer_size = case_dir['hidden_layer_size'] 
    num_hidden_layers = 5
    train_test = case_dir['split']
    epochs = case_dir['epochs'] 
    learning_rate = case_dir['learning_rate']
    forward = case_dir['forward']
    
    if start_time_test is not None:
        start_time = start_time_test
        
    if stop_time_test is not None:
        stop_time = stop_time_test
    

    loss_weights = [case_dir['l_bc'], case_dir['l_u'], case_dir['l_v'], case_dir['l_w']]
    
    results_directory = os.path.join(root, simulation_name)
    simulation_filepath = os.path.join(root, simulation_name)
    
    my_system = fk_system(start_time, stop_time, step_dt)

    observed, v, w, u, d, phi = my_system.read_simulation(simulation_filepath)

    geomtime = my_system.geomtime

    bc = my_system.BC_func(geomtime)

    all_data = np.hstack((observed, phi, d, v, w, u))

    criterion = RMSELoss()
    
    # Creating network
    input_size = 3
    output_size = 3

    net_ap = FNN_AP([input_size] + [hidden_layer_size] * num_hidden_layers + [output_size])
    
    train_set = egm_sampler(all_data, my_system)
    test_set = egm_sampler(all_data[:128*128], my_system)
    
    trainer = model_trainer_egm_homogeneous(net_ap, train_set, test_set, 64, geomtime, bc, criterion, my_system)
    
    model_files = ([f for f in os.listdir(os.path.join(root, case_name)) if 'pt' in f])
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(root, case_name,x)))


    trainer.model_u = torch.load(os.path.join(root, case_name, model_files[-1]))
 
    
    backend = matplotlib.get_backend()
    matplotlib.use("nbAgg")
    fig, ax = plt.subplots(2, 4, figsize=(10,5))
    
    train_set[0]
    
    vmin = 0
    vmax = 1
    
    def init():
        imgs = []
        phi_hat, u_hat, v_hat, w_hat = get_predicted_state(train_set[0], trainer)
        phi, u, v, w = get_gt_state(train_set[0], trainer)
        
        im = ax[0,0].imshow(phi_hat, animated=True, cmap="magma", vmin = np.min(phi), vmax = np.max(phi))
        ax[0,0].set_title('EGM prediction')
        plt.colorbar(im, ax = ax[0,0])
        imgs.append(im)
        
        im = ax[0,1].imshow(u_hat, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        ax[0,1].set_title('u prediction')
        plt.colorbar(im, ax = ax[0,1])
        imgs.append(im)
        
        im = ax[0,2].imshow(v_hat, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        ax[0,2].set_title('v prediction')
        plt.colorbar(im, ax = ax[0,2])
        imgs.append(im)
        
        im = ax[0,3].imshow(w_hat, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        ax[0,3].set_title('w prediction')
        plt.colorbar(im, ax = ax[0,3])
        imgs.append(im)
        
        im = ax[1,0].imshow(phi, animated=True, cmap="magma", vmin = np.min(phi), vmax = np.max(phi))
        ax[1,0].set_title('EGM GT')
        plt.colorbar(im, ax = ax[1,0])
        imgs.append(im)
        
        im = ax[1,1].imshow(u, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        ax[1,1].set_title('u GT')
        plt.colorbar(im, ax = ax[1,1])
        imgs.append(im)
        
        im = ax[1,2].imshow(v, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        ax[1,2].set_title('v GT')
        plt.colorbar(im, ax = ax[1,2])
        imgs.append(im)
        
        im = ax[1,3].imshow(w, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        ax[1,3].set_title('w GT')
        plt.colorbar(im, ax = ax[1,3])
        imgs.append(im)
        
        
        return imgs
    
    def update(iteration):
        
        phi_hat, u_hat, v_hat, w_hat = get_predicted_state(train_set[iteration], trainer)
        phi, u, v, w = get_gt_state(train_set[iteration], trainer)
        
        im = ax[0,0].imshow(phi_hat, animated=True, cmap="magma")
        
        im = ax[0,1].imshow(u_hat, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        
        im = ax[0,2].imshow(v_hat, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        
        im = ax[0,3].imshow(w_hat, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        
        im = ax[1,0].imshow(phi, animated=True, cmap="magma")
        
        im = ax[1,1].imshow(u, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        
        im = ax[1,2].imshow(v, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        
        im = ax[1,3].imshow(w, animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        ax[0,2].set_title("t: %d" % iteration)
        
        return ax
    animation = FuncAnimation(fig, update, frames=range(len(train_set)), init_func=init, blit=True)
    if filename is not None:
        writergif = PillowWriter(fps = 10)
        animation.save(filename, writer=writergif)
    
    matplotlib.use(backend)
    return HTML(animation.to_html5_video())
    