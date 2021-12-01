from egm_pinns import *
import argparse
import os
import pickle
from sklearn.model_selection import train_test_split

# python train.py --simulation pinn_egm_1a_planar --epochs 100000 --l-bc -1 --l-u -1 --l-w -1 --l-v -1 --start 10 --end 160 --step 5  --split 0.3 --forward 0           ############### running remotely
# python train.py --simulation pinn_egm_1a_planar --epochs 100000 --l-bc -1 --l-u -1 --l-w -1 --l-v -1 --start 10 --end 160 --step 5  --split 0.3 --forward 1           ############### running locally
# python train.py --simulation pinn_egm_1a_planar --epochs 100000 --l-bc 0 --l-u 0 --l-w 0 --l-v 0 --start 10 --end 160 --step 5  --split 0.3 --forward 0               ############### running locally
# python train.py --simulation pinn_egm_1a_planar --epochs 100000 --l-bc 0 --l-u 0 --l-w 0 --l-v 0 --start 10 --end 160 --step 5  --split 0.3 --forward 1               ############### running locally
# python train.py --simulation pinn_egm_1a_planar --epochs 100000 --l-bc 100 --l-u 100 --l-w 100 --l-v 100 --start 10 --end 160 --step 5  --split 0.3 --forward 0       ############### running remotely
# python train.py --simulation pinn_egm_1a_planar --epochs 100000 --l-bc 100 --l-u 100 --l-w 100 --l-v 100 --start 10 --end 160 --step 5  --split 0.3 --forward 1
# python train.py --simulation pinn_egm_1a_planar --epochs 100000 --l-bc 1000 --l-u 1000 --l-w 1000 --l-v 1000 --start 10 --end 160 --step 5  --split 0.3 --forward 0   ############### DO NOT RUN - DIVERGE
# python train.py --simulation pinn_egm_1a_planar --epochs 100000 --l-bc 1000 --l-u 1000 --l-w 1000 --l-v 1000 --start 10 --end 160 --step 5  --split 0.3 --forward 1   ############### running locally - DIVERGE   

# python train.py --simulation pinn_egm_1a_circular --epochs 100000 --l-bc -1 --l-u -1 --l-w -1 --l-v -1 --start 10 --end 210 --step 5  --split 0.3 --forward 0
# python train.py --simulation pinn_egm_1a_circular --epochs 100000 --l-bc -1 --l-u -1 --l-w -1 --l-v -1 --start 10 --end 210 --step 5  --split 0.3 --forward 1
# python train.py --simulation pinn_egm_1a_circular --epochs 100000 --l-bc 0 --l-u 0 --l-w 0 --l-v 0 --start 10 --end 210 --step 5  --split 0.3 --forward 0
# python train.py --simulation pinn_egm_1a_circular --epochs 100000 --l-bc 0 --l-u 0 --l-w 0 --l-v 0 --start 10 --end 210 --step 5  --split 0.3 --forward 1
# python train.py --simulation pinn_egm_1a_circular --epochs 100000 --l-bc 100 --l-u 100 --l-w 100 --l-v 100 --start 10 --end 210 --step 5  --split 0.3 --forward 0
# python train.py --simulation pinn_egm_1a_circular --epochs 100000 --l-bc 100 --l-u 100 --l-w 100 --l-v 100 --start 10 --end 210 --step 5  --split 0.3 --forward 1
# python train.py --simulation pinn_egm_1a_circular --epochs 100000 --l-bc 1000 --l-u 1000 --l-w 1000 --l-v 1000 --start 10 --end 210 --step 5  --split 0.3 --forward 0 ############### DO NOT RUN - DIVERGE
# python train.py --simulation pinn_egm_1a_circular --epochs 100000 --l-bc 1000 --l-u 1000 --l-w 1000 --l-v 1000 --start 10 --end 210 --step 5  --split 0.3 --forward 1 ############### DO NOT RUN - DIVERGE

# python train.py --simulation pinn_egm_1a_spiral --epochs 100000 --l-bc -1 --l-u -1 --l-w -1 --l-v -1 --start 250 --end 2250 --step 5  --split 0.3 --forward 0         ############### running remotely
# python train.py --simulation pinn_egm_1a_spiral --epochs 100000 --l-bc -1 --l-u -1 --l-w -1 --l-v -1 --start 250 --end 2250 --step 5  --split 0.3 --forward 1         ############### running remotely
# python train.py --simulation pinn_egm_1a_spiral --epochs 100000 --l-bc 0 --l-u 0 --l-w 0 --l-v 0 --start 250 --end 2250 --step 5  --split 0.3 --forward 0             ############### running locally
# python train.py --simulation pinn_egm_1a_spiral --epochs 100000 --l-bc 0 --l-u 0 --l-w 0 --l-v 0 --start 250 --end 2250 --step 5  --split 0.5 --forward 1             ############### running remotely
# python train.py --simulation pinn_egm_1a_spiral --epochs 100000 --l-bc 100 --l-u 100 --l-w 100 --l-v 100 --start 250 --end 2250 --step 5  --split 0.5 --forward 0
# python train.py --simulation pinn_egm_1a_spiral --epochs 100000 --l-bc 100 --l-u 100 --l-w 100 --l-v 100 --start 250 --end 2250 --step 5  --split 0.5 --forward 1     ############### running locally
# python train.py --simulation pinn_egm_1a_spiral --epochs 100000 --l-bc 1000 --l-u 1000 --l-w 1000 --l-v 1000 --start 250 --end 2250 --step 5  --split 0.5 --forward 0 ############### DO NOT RUN - DIVERGE
# python train.py --simulation pinn_egm_1a_spiral --epochs 100000 --l-bc 1000 --l-u 1000 --l-w 1000 --l-v 1000 --start 250 --end 2250 --step 5  --split 0.5 --forward 1 ############### DO NOT RUN - DIVERGE



parser = argparse.ArgumentParser()
parser.add_argument('-i', '--simulation', default='pinn_egm_1a_planar')
parser.add_argument('-s', '--start', default=0)
parser.add_argument('-f', '--end', default=100)
parser.add_argument('-d', '--step', default=1)
parser.add_argument('--split', default=0.8)
parser.add_argument('--hidden-layers', default=5)
parser.add_argument( '--size', default=60)
parser.add_argument('-b', '--batch', default=64)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--epochs', default=1000)
parser.add_argument('--forward', default=0)
parser.add_argument('--incremental', default=0)


parser.add_argument('--l-bc', default=0)
parser.add_argument('--l-u', default=0)
parser.add_argument('--l-v', default=0)
parser.add_argument('--l-w', default=0)




args = parser.parse_args()

if __name__ == '__main__':
    simulations_directory = '/home/konstantinos/Documents/PhD/fk_sims/two_d/pinn_egm/'

    simulation_name = str(args.simulation)+ '.hdf5'
    start_time = int(args.start)
    stop_time = int(args.end) 
    step_dt = int(args.step)
    hidden_layer_size = int(args.size)
    num_hidden_layers = int(args.hidden_layers)
    train_test = float(args.split)
    epochs = int(args.epochs)
    batch_size = int(args.batch)
    learning_rate = float(args.lr)
    forward = bool(int(args.forward))
    incremental = bool(int(args.incremental))

    loss_weights = [float(args.l_bc), float(args.l_u), float(args.l_v), float(args.l_w)]

    results_directory = f'training_{args.simulation}_{start_time}_{stop_time}_{step_dt}_{hidden_layer_size}_{learning_rate}_{int(train_test*10)}_{epochs}_{int(args.l_bc)}_{int(args.l_u)}_{int(args.l_v)}_{int(args.l_w)}'
    if forward:
        results_directory = 'forward_' + results_directory

    results_directory = os.path.join(simulations_directory, results_directory)
    os.makedirs(results_directory, exist_ok = True)

    simulation_filepath = os.path.join(simulations_directory, simulation_name)

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

    if forward == True:
        idx_for_train_test_split = int( train_test * len(all_data)//(128*128))

        train_set = all_data[:128*128*idx_for_train_test_split]
        test_set = all_data[128*128*idx_for_train_test_split:]

        train_set = egm_sampler(train_set, my_system)
        test_set = egm_sampler(test_set, my_system)
    else:
        my_dataset = egm_sampler(all_data, my_system)
        train_set, test_set = train_test_split(my_dataset, train_size = train_test, random_state = 404 )


    trainer = model_trainer_egm_homogeneous(net_ap, train_set, test_set, 64, geomtime, bc, criterion, my_system)

    if (forward == True) & (incremental == True):
        model, losses = trainer.incremental_train(epochs, learning_rate, False, results_directory, loss_weights)
    
    else:
        model, losses = trainer.train(epochs, learning_rate, True, results_directory, loss_weights)

    torch.save(model, os.path.join(results_directory, 'final_model.pt'))

    with open(os.path.join(results_directory, 'losses.pickle'), 'wb') as handle:
        pickle.dump(losses, handle)





