import numpy as np
import swiglpk as glpk
from dotmap import DotMap
 
def linprog(c_miniumize_direction, A_ub, b_ub, bounds=(0, np.inf)):
    """linprog-like interface with GLPK"""

    print(f"linprog_glpk.py: linprog called with bounds: {bounds}")

    A_ub = np.asarray(A_ub, dtype=float)
    b_ub = np.asarray(b_ub, dtype=float)
    c_miniumize_direction = np.asarray(c_miniumize_direction, dtype=float)

    m, n = A_ub.shape # m is number of constraints, n is number of variables
    assert b_ub.shape == (m,)
    assert c_miniumize_direction.shape == (n,)

    lp = glpk.glp_create_prob()

    # Add rows (constraints) with upper bounds: a_i^T x <= b_i
    glpk.glp_add_rows(lp, m)

    for i in range(m):
        glpk.glp_set_row_bnds(lp, i + 1, glpk.GLP_UP, 0.0, float(b_ub[i]))


    # bounds can be either a pair of numbers or a list of pairs of numbers
    if isinstance(bounds, tuple):
        bounds = [bounds] * n

    assert len(bounds) == n, f"bounds must be a list of {n} pairs of numbers"

    # Add columns (variables)
    glpk.glp_add_cols(lp, n)
    for j in range(n):

        lb, ub = bounds[j]

        if lb is None or np.isneginf(lb):
            lb = -np.inf
        if ub is None or np.isposinf(ub):
            ub = np.inf

        #print(f"{j=},lb: {lb}, ub: {ub}")

        if lb == -np.inf and ub == np.inf:
            glpk.glp_set_col_bnds(lp, j + 1, glpk.GLP_FR, 0.0, 0.0)
        elif lb == -np.inf:
            glpk.glp_set_col_bnds(lp, j + 1, glpk.GLP_UP, 0.0, float(ub))
        elif ub == np.inf:
            glpk.glp_set_col_bnds(lp, j + 1, glpk.GLP_LO, float(lb), 0.0)
        else:
            glpk.glp_set_col_bnds(lp, j + 1, glpk.GLP_DB, float(lb), float(ub))


    # set objective
    for j in range(n):
        glpk.glp_set_obj_coef(lp, j + 1, float(c_miniumize_direction[j]))

    # Fill the constraint matrix row-by-row
    # GLPK requires setting all the non-zero elements
    indices = glpk.intArray(n + 1)     # note: glpk indices are offset by 1 (first element is at position 1)
    values = glpk.doubleArray(n + 1)
    
    for row in range(m):
        cur_index = 1

        for col in range(n):
            value = A_ub[row, col]

            if value != 0.0:
                indices[cur_index] = col + 1
                values[cur_index] = float(value)
                cur_index += 1

        num_nonzeros = cur_index - 1

        if num_nonzeros > 0:
            glpk.glp_set_mat_row(lp, row + 1, num_nonzeros, indices, values)

    # set solver parameters (for example, reduce printing to only error messages)
    params = glpk.glp_smcp()
    glpk.glp_init_smcp(params)

    params.msg_lev = glpk.GLP_MSG_ERR # don't print status messages

    # 30 second time limit
    params.tm_lim = 30

    #params.msg_lev = glpk.GLP_MSG_ALL
    
    #params.presolve = glpk.GLP_ON
    #glpk.glp_scale_prob(lp, glpk.GLP_SF_AUTO)   # auto scaling
    #glpk.glp_std_basis(lp)                      # safe starting basis

    # Solve using the simplex algorithm
    simplex_res = glpk.glp_simplex(lp, params)
    rv = DotMap()

    if simplex_res != 0:
        rv.success = False
        rv.message = f'Simplex solver failed with error code {simplex_res}. '

        if simplex_res == glpk.GLP_ETMLIM:
            rv.message += " (GLP_ETMLIM - time limit exceeded)"
        elif simplex_res == glpk.GLP_ENOPFS:
            rv.message += " (GLP_ENOPFS - no feasible solution found)"
        elif simplex_res == glpk.GLP_ENODFS:
            rv.message += " (GLP_ENODFS - no dual feasible solution found)"
        elif simplex_res == glpk.GLP_EFAIL:
            rv.message += " (GLP_EFAIL - solver failed)"
        else:
            rv.message += f" (unknown error code {simplex_res})"
    else:
        # Verify optimality
        status = glpk.glp_get_status(lp)

        if status != glpk.GLP_OPT:
            rv.success = False
            rv.message = f'Optimization failed with status code {status}.'
        else:
            rv.success = True
            rv.message = 'Optimization successful.'        

            # Extract solution
            rv.x = np.array([glpk.glp_get_col_prim(lp, j + 1) for j in range(n)], dtype=float)
    
    glpk.glp_delete_prob(lp)

    return rv
