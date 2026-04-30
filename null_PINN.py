import spacepy.pycdf as pycdf
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, jacfwd
import optax
import time
from scipy.optimize import root
from multiprocessing import Pool, cpu_count

# =========================
# PARALLEL DATA LOADING
# =========================
start = time.time()
def process_mms(args):
    k, global_start = args

    file = f"D:/PhD courses/ML/Project/MMS_data/mms{k}_fgm_brst_l2_20150919074024_v4.18.0.cdf"
    data = pycdf.CDF(file)

    b_vals = np.array(data[f"mms{k}_fgm_b_gse_brst_l2"])[:, :3]
    r_vals = np.array(data[f"mms{k}_fgm_r_gse_brst_l2"])[:, :3]

    t_b = np.array(data["Epoch"])
    t_r = np.array(data["Epoch_state"])

    t_b_float = np.array([(tb - global_start).total_seconds() for tb in t_b])
    t_r_float = np.array([(tr - global_start).total_seconds() for tr in t_r])

    r_sync = np.vstack([
        np.interp(t_b_float, t_r_float, r_vals[:, i])
        for i in range(3)
    ]).T

    return r_sync, b_vals, t_b_float


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    

    # Reference time
    file0 = "D:/PhD courses/ML/Project/MMS_data/mms1_fgm_brst_l2_20150919074024_v4.18.0.cdf"
    data0 = pycdf.CDF(file0)
    global_start = np.array(data0['Epoch'])[0]

    # Parallel read
    with Pool(min(4, cpu_count())) as pool:
        results = pool.map(process_mms, [(k, global_start) for k in range(1,5)])

    R, B, T = zip(*results)

    # =========================
    # TIME ALIGNMENT
    # =========================

    t_ref = T[0]
    B_interp = []

    for k in range(4):
        if k == 0:
            B_interp.append(B[0])
        else:
            Bk_interp = np.vstack([
                np.interp(t_ref, T[k], B[k][:, i])
                for i in range(3)
            ]).T
            B_interp.append(Bk_interp)

    t = t_ref
    mask = (t >=180 ) & (t <= 240 )
    print("mask = ", mask)

    R_masked = [R[k][mask] for k in range(4)]
    B_masked = [B_interp[k][mask] for k in range(4)]
    t_masked = t[mask]
    # =========================
    # TRAIN / TEST
    # =========================

    r_train = np.vstack(R_masked[:3])
    b_train = np.vstack(B_masked[:3])

    R_test = R_masked[3]
    B_test = B_masked[3]

    # =========================
    # NORMALIZATION
    # =========================

    r_min, r_max = r_train.min(0), r_train.max(0)
    b_min, b_max = b_train.min(0), b_train.max(0)

    def normalize(x, xmin, xmax):
        return 2*(x - xmin)/(xmax - xmin) - 1

    def denormalize(x, xmin, xmax):
        return (x + 1)/2 * (xmax - xmin) + xmin

    R_train = jnp.array(normalize(r_train, r_min, r_max))
    B_train = jnp.array(normalize(b_train, b_min, b_max))

    R_test_jax = jnp.array(normalize(R_test, r_min, r_max))
    B_test_jax = jnp.array(normalize(B_test, b_min, b_max))

    # =========================
    # MODEL
    # =========================

    def init_mlp(key, layers):
        keys = random.split(key, len(layers)-1)
        params = []
        for k, (m, n) in zip(keys, zip(layers[:-1], layers[1:])):
            W = random.normal(k, (m, n)) * jnp.sqrt(2/m)
            b = jnp.zeros((n,))
            params.append((W, b))
        return params

    def forward(params, x):
        for W, b in params[:-1]:
            x = jnp.tanh(x @ W + b)
        W, b = params[-1]
        return x @ W + b

    forward_batch = vmap(forward, in_axes=(None, 0))

    # =========================
    # PHYSICS 
    # =========================

    def compute_physics(params, X):

        def B_single(x):
            return forward(params, x)

        # Forward-mode Jacobian (vectorized)
        J = vmap(jacfwd(B_single))(X)   # (N,3,3)

        B = forward_batch(params, X)

        divB = jnp.trace(J, axis1=1, axis2=2)

        curl = jnp.stack([
            J[:,2,1] - J[:,1,2],
            J[:,0,2] - J[:,2,0],
            J[:,1,0] - J[:,0,1]
        ], axis=1)

        JxB = jnp.cross(curl, B)

        return divB, JxB

    # =========================
    # LOSS
    # =========================

    @jit
    def loss_fn(params, R_data, B_data, R_col):

        X_all = jnp.vstack([R_data, R_col])

        div_all, jxb_all = compute_physics(params, X_all)
        B_pred = forward_batch(params, R_data)

        N = R_data.shape[0]

        div_d, div_c = div_all[:N], div_all[N:]
        jxb_d, jxb_c = jxb_all[:N], jxb_all[N:]

        loss_data = jnp.mean((B_pred - B_data)**2)
        loss_div = jnp.mean(div_d**2) + jnp.mean(div_c**2)
        loss_jxb = jnp.mean(jxb_d**2) + jnp.mean(jxb_c**2)

        return loss_data + 0.1*loss_div + 0.1*loss_jxb

    @jit
    def train_step(params, opt_state, R_data, B_data, R_col):

        loss, grads = jax.value_and_grad(loss_fn)(params, R_data, B_data, R_col)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    # =========================
    # TRAINING
    # =========================
    point_1 = time.time()
    key = random.PRNGKey(0)
    params = init_mlp(key, [3,64,64,64,3])

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    N_col = 2000

    for epoch in range(5001):

        if epoch % 500 == 0:
            key, subkey = random.split(key)
            R_col = random.uniform(subkey, (N_col,3), minval=-1, maxval=1)

        params, opt_state, loss = train_step(
            params, opt_state,
            R_train, B_train,
            R_col
        )

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss = {loss:.5e}")
    point_2 = time.time()
    # =========================
    # TEST
    # =========================

    @jit
    def predict(params, R):
        return forward_batch(params, R)

    B_pred = predict(params, R_test_jax)
    print("Test Error =", jnp.mean((B_pred - B_test_jax)**2))

    # =========================
    # NULL FINDING
    # =========================
    R_barycenter = (R_masked[0] + R_masked[1] + R_masked[2] + R_masked[3]) / 4
    
    
    def magnetic_field_root(x):
        return np.array(forward(params, jnp.array(x)))

    def magnetic_jacobian(x):
        return np.array(jacfwd(lambda x: forward(params, x))(jnp.array(x)))

    guesses = [np.zeros(3), np.array([0.5,0,0]), np.array([-0.5,0,0]),np.array([0,0.5,0])]

    for x0 in guesses:
        sol = root(magnetic_field_root, x0, jac=magnetic_jacobian)

        if sol.success:
            null = denormalize(sol.x, r_min, r_max)
            print("\nNull:", null, "|B|=", np.linalg.norm(sol.fun))
            distances = np.linalg.norm(R_barycenter - null, axis=1)

            idx_min = np.argmin(distances)
            
            # 4. Get the time from your masked time array
            null_time_seconds = t_masked[idx_min]
            sat_pos = np.array([R_masked[0][idx_min], R_masked[1][idx_min], R_masked[2][idx_min], R_masked[3][idx_min]])
            print('sat_pos =' ,sat_pos)

            print(f"The Null Point was most active/closest at: {null_time_seconds:.2f} seconds into the interval.")
        
        
        else:
            print("Root finding failed to converge.")
    print("Runtime =", time.time() - start)
    print("training time = ", point_2 -start)
    
    
    
    
    



