import os
import numpy as np

from compare_util import load_data_and_sim_f16, make_linear_predictor, load_data_and_sim_f16_turns
from run_compare_dubins import hash_file
from aerobench.util import get_script_path
from scipy.io import savemat

def load(psidot_max, num_steps_at_max_turn):
    '''load data'''

    np.set_printoptions(suppress=True, precision=3)

    dubins_filename = 'dubins_data.pkl'

    script_dir = get_script_path(__file__)
    dubins_file_path = os.path.join(script_dir, dubins_filename)

    file_hash = hash_file(dubins_file_path)
    print(f"SHA-256 hash of {dubins_filename}: {file_hash}")

    MAX_INDEX = 20
    odd_indices = tuple(range(1, MAX_INDEX, 2))
    even_indices = tuple(range(0, MAX_INDEX, 2))

    lead_res_dicts = load_data_and_sim_f16(dubins_file_path, file_hash, single_index=odd_indices)
    wingman_res_dicts = load_data_and_sim_f16(dubins_file_path, file_hash, single_index=even_indices)

    
    wingman_right_res_dicts = load_data_and_sim_f16_turns(dubins_file_path, file_hash, even_indices, psidot_max, num_steps_at_max_turn)
    wingman_left_res_dicts = load_data_and_sim_f16_turns(dubins_file_path, file_hash, even_indices, -psidot_max, num_steps_at_max_turn)


    return lead_res_dicts, wingman_res_dicts, wingman_right_res_dicts, wingman_left_res_dicts

def main():
    '''main entry point'''
    
    MAX_STEPS_TO_PREDICT = 5
    psidot_max = 10*np.pi/180
    lead_res_dicts, wingman_res_dicts, wingman_right_res_dicts, wingman_left_res_dicts = load(psidot_max, MAX_STEPS_TO_PREDICT)

    for label, res_dicts, trim_to_max_steps in [('lead', lead_res_dicts, False), ('wingman', wingman_res_dicts, False), 
                             ('wingman_right', wingman_right_res_dicts, True), ('wingman_left', wingman_left_res_dicts, True)]:
        print("------------")
        print(f'--- {label} ---')
        predictor = make_linear_predictor(res_dicts, MAX_STEPS_TO_PREDICT, trim_to_max_steps)
           
        data = {'max_steps': MAX_STEPS_TO_PREDICT}

        for step in range(1, MAX_STEPS_TO_PREDICT+1):
            A = predictor.A_dict[step]
            residuals = predictor.residuals_dict[step]
            data[f'A_{step}'] = A
            data[f'residuals_{step}'] = residuals

        filename = f'{label}_predictor.mat'
        savemat(filename, data)
        print(f"Saved {label} predictor A matrices and residuals to {filename}")


if __name__ == '__main__':
    main() 
