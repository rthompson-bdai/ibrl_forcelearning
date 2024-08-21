
import argparse
import h5py
import numpy as np
import os


def norm_file(file, normalization_file, output_name):
    #load normalization info
    f = h5py.File(normalization_file, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    force_data  = []
    torque_data = []
    for demo_name in demos:
        demo = f["data/{}/obs".format(demo_name)]
        force_data.append(demo['robot0_ee_force'])
        torque_data.append(demo['robot0_ee_torque'])

    force_data = np.concatenate(force_data, 0)
    torque_data = np.concatenate(torque_data, 0)
    force_mean = np.mean(force_data, axis=0)
    torque_mean = np.mean(torque_data, axis=0)
    force_std= np.std(force_data, axis=0)
    torque_std = np.std(torque_data, axis=0)


    #load and iterate over file

    force_f = h5py.File(file, "r")
    force_demos = list(force_f["data"].keys())
    force_inds = np.argsort([int(elem[5:]) for elem in force_demos])
    force_demos = [force_demos[i] for i in force_inds]

    output_path = os.path.join(os.path.dirname(normalization_file), output_name)
    f_out = h5py.File(output_path, "w")
    #data_grp = f_out.create_group("data")

    force_f.copy(force_f['data'], f_out, 'data')

    total_samples = 0
    for ind in range(len(force_demos)):
        ep = f_out["data"][force_demos[ind]]
        #print(ep['demo_0'].keys())
        force_data = (ep['obs']['robot0_ee_force'][:,:] - force_mean) / force_std
        torque_data = (ep['obs']['robot0_ee_torque'][:,:] - torque_mean) / torque_std

        ep['obs']['robot0_ee_force'][:,:] = force_data #vfunc_force(force_data)
        ep['obs']['robot0_ee_torque'][:,:] = torque_data #vfunc_torque(torque_data)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--norm_dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    norm_file(args.file, args.norm_dataset, args.output)