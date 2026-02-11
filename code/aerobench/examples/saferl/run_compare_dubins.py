'''load rollouts from safeRL and compare to f-16 recreations with same inputs

Get a predictive model for online RTA
'''


import os
import hashlib
import numpy as np
from scipy.io import savemat, loadmat

from compare_util import load_data_and_sim_f16, eval_predictor_accuracy, input_obs_func, output_obs_func, make_linear_predictor, plot_prediction_vs_dubins, RegressionPredictor, plot_single_prediction
from aerobench.util import get_script_path, StateIndex
from matplotlib import pyplot as plt

def hash_file(filename, algorithm="sha256"):
    """Compute the hash of a file using the specified algorithm (default: SHA-256)."""
    hash_func = getattr(hashlib, algorithm)()
    with open(filename, "rb") as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def main():
    '''main function'''

    np.set_printoptions(suppress=True, precision=3)

    dubins_filename = 'dubins_data.pkl'

    script_dir = get_script_path(__file__)
    dubins_file_path = os.path.join(script_dir, dubins_filename)

    file_hash = hash_file(dubins_file_path)
    print(f"SHA-256 hash of {dubins_filename}: {file_hash}")

    all_f16_res_dicts = load_data_and_sim_f16(dubins_file_path, file_hash, single_index=None)

    MAX_STEPS_TO_PREDICT = 19

    filename = 'linear_predictor.mat'

    # if predictor exists, just load it
    if os.path.exists(filename):
        data = loadmat(filename)
        max_steps = data['max_steps'][0, 0]

        assert max_steps == MAX_STEPS_TO_PREDICT, f"max_steps {max_steps} != {MAX_STEPS_TO_PREDICT}"
        print(f"Loaded existing predictor from {filename} with max_steps: {MAX_STEPS_TO_PREDICT}")

        # create predictor object from data
        predictor = RegressionPredictor(input_obs_func, output_obs_func)
        for step in range(1, MAX_STEPS_TO_PREDICT+1):
            A = data[f'A_{step}']
            residuals = data[f'residuals_{step}']
            predictor.A_dict[step] = A
            predictor.residuals_dict[step] = residuals

        print(f"Loaded predictor from {filename}")
    else:
        print(f"Creating new predictor from scratch and saving to {filename}")
        predictor = make_linear_predictor(all_f16_res_dicts, MAX_STEPS_TO_PREDICT)
        
        data = {'max_steps': MAX_STEPS_TO_PREDICT}

        for step in range(1, MAX_STEPS_TO_PREDICT+1):
            A = predictor.A_dict[step]
            residuals = predictor.residuals_dict[step]
            data[f'A_{step}'] = A
            data[f'residuals_{step}'] = residuals

        savemat(filename, data)
        print(f"Saved predictor A matrices and residuals to {filename}")

    # plot single sim with boxes
    plot_single_prediction(traj_index=33, num_steps=11, predictor=predictor, all_f16_res_dicts=all_f16_res_dicts)

    #########

    #for num_steps_to_predict in range(1, MAX_STEPS_TO_PREDICT+1):
    for num_steps_to_predict in [9]: #range(1, 10+1):
        
        # print residual
        resid = predictor.residuals_dict[num_steps_to_predict] 
        print(f"\nResiduals for step {num_steps_to_predict}: {resid}")

        for num_trajectories in [400, 200, 100]:
            max_error_vector = eval_predictor_accuracy(all_f16_res_dicts, predictor, num_steps_to_predict=num_steps_to_predict, num_trajectories=num_trajectories)
            print(f"[{num_trajectories=}] observed max error at {num_steps_to_predict=}: {max_error_vector=}")

    # one plot for all steps
    plot_prediction_vs_dubins(all_f16_res_dicts, predictor, num_steps_to_predict=MAX_STEPS_TO_PREDICT)

    #compare_dubins_data() # create data file from f-16 recreations of dubins paths=

if __name__ == '__main__':
    main()
