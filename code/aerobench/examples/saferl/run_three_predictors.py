import os
import numpy as np

from compare_util import load_data_and_sim_f16, make_linear_predictor, load_data_and_sim_f16_turns
from run_compare_dubins import hash_file
from aerobench.util import get_script_path
from scipy.io import savemat, loadmat

STEP_DELTA = 0.25
MAX_STEPS_TO_PREDICT = 5

def load(psidot_max, num_steps_at_max_turn, step_delta=1.0):
    '''load data'''

    np.set_printoptions(suppress=True, precision=3)

    dubins_filename = 'dubins_data.pkl'

    script_dir = get_script_path(__file__)
    dubins_file_path = os.path.join(script_dir, dubins_filename)

    file_hash = hash_file(dubins_file_path)
    print(f"SHA-256 hash of {dubins_filename}: {file_hash}")

    MAX_INDEX = 400
    odd_indices = tuple(range(1, MAX_INDEX, 2))
    even_indices = tuple(range(0, MAX_INDEX, 2))

    lead_res_dicts = load_data_and_sim_f16(dubins_file_path, file_hash, single_index=odd_indices, step_delta=step_delta)
    wingman_res_dicts = load_data_and_sim_f16(dubins_file_path, file_hash, single_index=even_indices, step_delta=step_delta)

    wingman_right_res_dicts = load_data_and_sim_f16_turns(dubins_file_path, file_hash, even_indices, psidot_max, num_steps_at_max_turn, step_delta=step_delta)
    wingman_left_res_dicts = load_data_and_sim_f16_turns(dubins_file_path, file_hash, even_indices, -psidot_max, num_steps_at_max_turn, step_delta=step_delta)

    return lead_res_dicts, wingman_res_dicts, wingman_right_res_dicts, wingman_left_res_dicts

def main():
    '''main entry point'''

    psidot_max = 10*np.pi/180
    script_dir = get_script_path(__file__)

    # Print residuals from existing .mat files before retraining
    old_residuals = {}
    print("=== Current .mat predictor residuals ===")
    for label in ['lead', 'wingman', 'wingman_right', 'wingman_left']:
        mat_path = os.path.join(script_dir, f'{label}_predictor.mat')
        if os.path.exists(mat_path):
            mat = loadmat(mat_path)
            max_s = int(mat['max_steps'].flat[0])
            old_residuals[label] = {}
            print(f"--- {label} ---")
            for step in range(1, max_s + 1):
                r = mat[f'residuals_{step}'].flatten()
                old_residuals[label][step] = r
                print(f"  step {step} (t={float(step):.1f}s): {r}")

    lead_res_dicts, wingman_res_dicts, wingman_right_res_dicts, wingman_left_res_dicts = load(psidot_max, MAX_STEPS_TO_PREDICT, step_delta=STEP_DELTA)

    total_sim_steps = round(MAX_STEPS_TO_PREDICT / STEP_DELTA)

    for label, res_dicts, trim_to_max_steps, turn_cmd in [('lead', lead_res_dicts, False, 0.0), ('wingman', wingman_res_dicts, False, -999),
                             ('wingman_right', wingman_right_res_dicts, True, psidot_max), ('wingman_left', wingman_left_res_dicts, True, -psidot_max)]:
        print("------------")
        print(f'--- {label} ---')
        predictor = make_linear_predictor(res_dicts, MAX_STEPS_TO_PREDICT, trim_to_max_steps, step_delta=STEP_DELTA)

        data = {'max_steps': MAX_STEPS_TO_PREDICT, 'step_delta': STEP_DELTA, 'turn_cmd': turn_cmd}

        print(f"\n=== New residuals for {label} ===")
        for sim_step in range(1, total_sim_steps + 1):
            step_time = round(sim_step * STEP_DELTA, 10)
            step_str = str(step_time)           # "0.25", "0.5", ..., "5.0"
            mat_key = step_str.replace('.', '_')  # "0_25", "0_5", ..., "5_0"
            A = predictor.A_dict[sim_step]
            residuals = predictor.residuals_dict[sim_step]
            data[f'A_{mat_key}'] = A
            data[f'residuals_{mat_key}'] = residuals
            print(f"  t={step_str}s: residuals={residuals}")

        # Compare new residuals vs old at integer steps
        print(f"\n=== Comparison vs old .mat (integer steps) for {label} ===")
        for old_step in range(1, MAX_STEPS_TO_PREDICT + 1):
            sim_step = round(old_step / STEP_DELTA)
            new_r = predictor.residuals_dict[sim_step]
            if label in old_residuals and old_step in old_residuals[label]:
                old_r = old_residuals[label][old_step]
                match = "~MATCH" if np.allclose(old_r, new_r, rtol=0.1) else "DIFF"
                print(f"  t={old_step}s: old={old_r}  new={new_r}  {match}")
            else:
                print(f"  t={old_step}s: new={new_r}")

        filename = f'{label}_predictor.mat'
        savemat(filename, data)
        print(f"Saved {label} predictor to {filename}")


if __name__ == '__main__':
    main()
