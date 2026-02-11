#!/usr/bin/env python3
"""
Script to plot prediction lines and boxes for 1 to 10 steps using initial state
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.io import loadmat
import os
import sys

# Add parent directory to path to import aerobench modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from aerobench.util import StateIndex
from compare_util import WingmanF16State


def main():
    """Main function to plot predictions"""
    
    # First initial state (16-dimensional state vector)
    state16_1 = np.array([
        766.4,  # vt
        0, #0.037,#0,         # alpha (0.037 / 0)
        0,         # beta
        0,         # phi
        0,         # theta
        -2.8, # psi
        0,         # P
        0,         # Q
        0,         # R
        0,         # pn
        0,         # pe
        1000,  # h (altitude)
        9,  # pow
        0,         # Nz
        0,         # Ny
        0          # Nx
    ])
    
    # Target values for first state
    psi_target_1 = -2.8251
    vel_target_1 = 766.3860
    
    # Second initial state (16-dimensional state vector)
    state16_2 = np.array([
        766.4,#785.759,    # vt
        0.037,      # alpha
        0.,         # beta
        0.,         # phi
        0.,         # theta
        -1.878,     # psi
        0.,         # P
        0.,         # Q
        0.,         # R
        0, #7487.181,   # pn
        0,#7264.726,   # pe
        1000.,      # h (altitude)
        9.,         # pow
        0.,         # Nz
        0.,         # Ny
        0.          # Nx
    ])
    
    # Target values for second state
    psi_target_2 = state16_2[StateIndex.PSI]#-1.8775006174630167
    vel_target_2 = state16_2[StateIndex.VT]#785.759468318621
    
    # Actions (2x11 array of zeros)
    actions = np.zeros((2, 11))
    
    # Load the linear predictor
    predictor_path = '/home/stan/repositories/AeroBenchVVPython/code/aerobench/linear_predictor.mat'
    predictor_data = loadmat(predictor_path)
    
    print(f"Loaded predictor with max_steps: {predictor_data['max_steps'][0, 0]}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define lists to store both predictions
    predictions = []
    
    # Process both initial states
    for state_idx, (state16, psi_target, vel_target, state_label, marker_style, line_style) in enumerate([
        (state16_1, psi_target_1, vel_target_1, "State 1 (Original)", 'o', '-'),
        (state16_2, psi_target_2, vel_target_2, "State 2 (New)", 's', '--')
    ]):
        # Extract x, y, heading, v from the 16-state vector
        x0 = state16[StateIndex.POS_E]  # pe (east position)
        y0 = state16[StateIndex.POS_N]  # pn (north position) 
        psi0 = state16[StateIndex.PSI]  # psi (yaw angle)
        v0 = state16[StateIndex.VT]     # vt (velocity)
        alt0 = state16[StateIndex.ALT]  # h (altitude)
        
        # Convert psi to heading
        heading0 = np.pi / 2 - psi0
        
        print(f"\n{state_label} initial state: x={x0:.1f}, y={y0:.1f}, heading={heading0:.4f}, v={v0:.1f}, alt={alt0:.1f}")
        
        # Plot initial position
        ax.plot(x0, y0, 'k' + marker_style, markersize=10, label=f'{state_label} Initial Position')
        
        # Color scheme for different prediction steps
        if state_idx == 0:
            colors = plt.cm.viridis(np.linspace(0, 1, 10))
        else:
            colors = plt.cm.plasma(np.linspace(0, 1, 10))
        
        # For each prediction step from 1 to 10
        for num_steps in range(1, 11):
            # Get A matrix and residuals for this number of steps
            A_key = f'A_{num_steps}'
            residuals_key = f'residuals_{num_steps}'
            
            if A_key not in predictor_data:
                print(f"Warning: No predictor for {num_steps} steps")
                continue
            
            A = predictor_data[A_key]
            residuals = predictor_data[residuals_key].flatten()
            
            # Create input observation vector
            # Based on the input_obs_func in compare_util.py
            dubins_state = np.array([x0, y0, heading0, v0, v0*np.cos(heading0), v0*np.sin(heading0)])
            
            # Full F16 state for input - use all 16 elements
            f16_state_13d = state16  # All 16 elements
            
            # Build input observation list following the exact format from compare_util.py
            input_obs_list = []
            
            # Add dubins state
            input_obs_list.append(dubins_state)
            
            # Add F16 state (16D)
            input_obs_list.append(f16_state_13d)
            
            # Initialize rollout states
            dubins_rollout_cur_state = dubins_state.copy()
            ideal_dubins_rollout_cur_state = dubins_state.copy()
            
            # Set ideal targets
            ideal_dubins_rollout_cur_state[3] = vel_target  # Update velocity
            ideal_dubins_rollout_cur_state[2] = np.pi / 2 - psi_target  # Update heading
            ideal_dubins_rollout_cur_state[4] = vel_target * np.cos(ideal_dubins_rollout_cur_state[2])
            ideal_dubins_rollout_cur_state[5] = vel_target * np.sin(ideal_dubins_rollout_cur_state[2])
            
            # For each step, add the action and update states
            for step in range(num_steps):
                action = actions[:, min(step, actions.shape[1]-1)]
                input_obs_list.append(action)
                
                # Update dubins rollout state
                next_state = dubins_rollout_cur_state.copy()
                next_state[0] = dubins_rollout_cur_state[0] + dubins_rollout_cur_state[3] * np.cos(dubins_rollout_cur_state[2])
                next_state[1] = dubins_rollout_cur_state[1] + dubins_rollout_cur_state[3] * np.sin(dubins_rollout_cur_state[2])
                next_state[2] = dubins_rollout_cur_state[2] + action[0]  # update theta
                next_state[3] = dubins_rollout_cur_state[3] + action[1]  # update vel
                next_state[4] = next_state[3] * np.cos(next_state[2])
                next_state[5] = next_state[3] * np.sin(next_state[2])
                
                # Update ideal dubins state
                next_ideal_state = ideal_dubins_rollout_cur_state.copy()
                next_ideal_state[0] = ideal_dubins_rollout_cur_state[0] + ideal_dubins_rollout_cur_state[3] * np.cos(ideal_dubins_rollout_cur_state[2])
                next_ideal_state[1] = ideal_dubins_rollout_cur_state[1] + ideal_dubins_rollout_cur_state[3] * np.sin(ideal_dubins_rollout_cur_state[2])
                next_ideal_state[2] = ideal_dubins_rollout_cur_state[2] + action[0]  # update theta
                next_ideal_state[3] = ideal_dubins_rollout_cur_state[3] + action[1]  # update vel
                next_ideal_state[4] = next_ideal_state[3] * np.cos(next_ideal_state[2])
                next_ideal_state[5] = next_ideal_state[3] * np.sin(next_ideal_state[2])
                
                dubins_rollout_cur_state = next_state
                ideal_dubins_rollout_cur_state = next_ideal_state
                
                # Add both states (all steps are included since LAST_N_STEPS = np.inf)
                input_obs_list.append(dubins_rollout_cur_state)
                input_obs_list.append(ideal_dubins_rollout_cur_state)
            
            # Add constant 1 at the end
            input_obs_list.append([1])
            
            # Flatten input observation
            input_obs = np.concatenate(input_obs_list)
            
            
            # Predict output
            predicted_output = input_obs @ A.T
            
            # Extract predicted position
            predicted_x = predicted_output[0]
            predicted_y = predicted_output[1]
            
            # Plot prediction line from initial position to predicted position
            #ax.plot([x0, predicted_x], [y0, predicted_y], '-', color=colors[num_steps-1], 
                    #linewidth=2, label=f'{num_steps} steps')
            
            # Plot predicted position
            ax.plot(predicted_x, predicted_y, marker_style, color=colors[num_steps-1], markersize=8)
            
            # Add uncertainty box based on residuals
            if len(residuals) >= 2:
                # Use L-infinity norm residuals for x and y
                x_residual = residuals[0] if len(residuals) > 0 else 50
                y_residual = residuals[1] if len(residuals) > 1 else 50
                
                # Create rectangle patch for uncertainty box
                rect = patches.Rectangle((predicted_x - x_residual, predicted_y - y_residual),
                                       2 * x_residual, 2 * y_residual,
                                       linewidth=1, edgecolor=colors[num_steps-1],
                                       facecolor='none', alpha=0.5, linestyle=line_style)
                ax.add_patch(rect)
                
                # Add text label for step number (only for last step to avoid clutter)
                if num_steps == 10:
                    ax.text(predicted_x + x_residual + 20, predicted_y, f'{state_label}',
                           fontsize=10, color=colors[num_steps-1])
    
    # Set labels and title
    ax.set_xlabel('East Position (ft)', fontsize=12)
    ax.set_ylabel('North Position (ft)', fontsize=12)
    ax.set_title('F-16 Position Predictions with Uncertainty Boxes (1-10 steps) - Two Initial States', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add legend
    ax.legend(loc='best', fontsize=10)
    
    # Save plot
    plt.tight_layout()
    filename = 'step0_predictions.png'
    plt.savefig(filename, dpi=150)
    print(f"Saved plot to {filename}")
    
    # Also show the plot
    #plt.show()
    
    # Print prediction details
    print("\nPrediction Details:")
    print("State 1 (Original):")
    print(f"  Initial: x=0.0, y=0.0, psi=-2.8000, v=766.4")
    print(f"  Targets: psi_target=-2.8251, vel_target=766.3860")
    print("\nState 2 (New):")
    print(f"  Initial: x=7264.7, y=7487.2, psi=-1.8780, v=785.8")
    print(f"  Targets: psi_target=-1.8775, vel_target=785.7595")
    print(f"\nActions: all zeros (no control inputs for all 10 steps)")


if __name__ == '__main__':
    main()
