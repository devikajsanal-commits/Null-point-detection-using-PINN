import spacepy.pycdf as pycdf
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit, jacfwd 
import optax
import time 


start = time.time()

R = []
B = []
T = []

file = f"D:/PhD courses/ML/Project/MMS_data/mms1_fgm_brst_l2_20150919074024_v4.18.0.cdf"
data = pycdf.CDF(file)
global_start = np.array(data['Epoch'])[0]

for k in range(1,5):

    file = f"D:/PhD courses/ML/Project/MMS_data/mms{k}_fgm_brst_l2_20150919074024_v4.18.0.cdf"
    data = pycdf.CDF(file)

    b_vals = np.array(data[f"mms{k}_fgm_b_gse_brst_l2"])[:, :3]
    r_vals = np.array(data[f"mms{k}_fgm_r_gse_brst_l2"])[:, :3]

    t_b = np.array(data["Epoch"])
    t_r = np.array(data["Epoch_state"])

    # TIME CONVERSION
    t_b_float = np.array([(tb - global_start).total_seconds() for tb in t_b])
    t_r_float = np.array([(tr - global_start).total_seconds() for tr in t_r])

    r_sync = np.zeros((len(t_b), 3))

    for i in range(3):
        r_sync[:, i] = np.interp(t_b_float, t_r_float, r_vals[:, i])

    b_sync = b_vals

    R.append(r_sync)
    B.append(b_sync)
    T.append(t_b_float)

t_ref = T[0]

B_interp = []

for k in range(4):

    if k == 0:
        # MMS1 already on reference grid
        B_interp.append(B[0])
    else:
        Bk = B[k]
        tk = T[k]

        Bk_interp = np.zeros((len(t_ref), 3))

        for i in range(3):
            Bk_interp[:, i] = np.interp(t_ref, tk, Bk[:, i])

        B_interp.append(Bk_interp)


t = t_ref   # unified time axis

mask = (t >= 0) & (t <= 300)

R_masked = [R[k][mask] for k in range(4)]
B_masked = [B_interp[k][mask] for k in range(4)]
t_masked = t[mask]

    
print("Separation (km):", np.linalg.norm(R_masked[0][0] - R_masked[1][0]))



def solid_angle(b1, b2, b3, eps = 1e-10):
    # normalize
    b1 = b1 / (jnp.linalg.norm(b1) + eps)
    b2 = b2 / (jnp.linalg.norm(b2) + eps)
    b3 = b3 / (jnp.linalg.norm(b3) + eps)

    triple = jnp.dot(b1, jnp.cross(b2, b3))
    denom = 1.0 + jnp.dot(b1, b2) + jnp.dot(b2, b3) + jnp.dot(b3, b1)

    return 2.0 * jnp.arctan2(triple, denom)


def poincare_index(B1, B2, B3, B4):
    Omega123 = solid_angle(B1, B2, B3)
    Omega124 = solid_angle(B1, B2, B4)
    Omega134 = solid_angle(B1, B3, B4)
    Omega234 = solid_angle(B2, B3, B4)

    I = (Omega123 + Omega124 + Omega134 + Omega234) / (4 * jnp.pi)
    return I

I_all = []

for k in range(len(t_masked)):

    B1 = B_masked[0][k]
    B2 = B_masked[1][k]
    B3 = B_masked[2][k]
    B4 = B_masked[3][k]

    I = poincare_index(
        jnp.array(B1),
        jnp.array(B2),
        jnp.array(B3),
        jnp.array(B4)
    )
    
    I_all.append(float(I))
    
threshold = 0.5
null_indices = [k for k in range(len(I_all)) if abs(I_all[k]) > threshold]

print("Null candidates:", len(null_indices))


null_positions = []
null_times = []


def is_inside_tetrahedron(p, r1, r2, r3, r4):
    """
    Checks if point p is inside the tetrahedron formed by r1, r2, r3, r4
    using barycentric coordinates.
    """
    def tet_volume(a, b, c, d):
        return abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0

    v_total = tet_volume(r1, r2, r3, r4)
    if v_total < 1e-6: return False # Degenerate tetrahedron
    
    # Volumes of sub-tetrahedra formed by the point p
    v1 = tet_volume(p, r2, r3, r4)
    v2 = tet_volume(r1, p, r3, r4)
    v3 = tet_volume(r1, r2, p, r4)
    v4 = tet_volume(r1, r2, r3, p)
    
    # Sum of sub-volumes should equal total volume if inside
    #  use a small epsilon for numerical stability
    #return np.isclose(v_total, v1 + v2 + v3 + v4, rtol=1e-3)
    return np.isclose(v_total, v1 + v2 + v3 + v4, rtol=0.2)
final_nulls = []

for k in null_indices:
   
    # positions
    r1 = R_masked[0][k]
    r2 = R_masked[1][k]
    r3 = R_masked[2][k]
    r4 = R_masked[3][k]

    # magnetic field
    b1 = B_masked[0][k]
    b2 = B_masked[1][k]
    b3 = B_masked[2][k]
    b4 = B_masked[3][k]

    # barycenter
    r0 = np.mean([r1, r2, r3, r4], axis=0)
    b0 = np.mean([b1, b2, b3, b4], axis=0)

    # construct matrices
    Rmat = np.vstack([r2 - r1, r3 - r1, r4 - r1]).T
    Bmat = np.vstack([b2 - b1, b3 - b1, b4 - b1]).T

    # compute Jacobian
    J = Bmat @ np.linalg.pinv(Rmat)

    # stability check
    if np.linalg.cond(J) > 1e6:
        continue

    # compute null position
    r_null = r0 - np.linalg.pinv(J) @ b0

    # residual check
    residual = np.linalg.norm(b0 + J @ (r_null - r0))

    if residual > 1e-2:
        continue

    null_positions.append(r_null)
    null_times.append(t_ref[k])
    
    

    # Check if r_null is physically inside the spacecraft formation
    if is_inside_tetrahedron(r_null, r1, r2, r3, r4):
        final_nulls.append({
            'time': t_ref[k],
            'pos': r_null
        })

print(f"Verified Nulls inside tetrahedron: {len(final_nulls)}")

from sklearn.cluster import DBSCAN

if len(final_nulls) > 0:
    # Prepare data for clustering (X, Y, Z coordinates)
    X_coords = np.array([n['pos'] for n in final_nulls])
    
    # eps is the max distance between two points to be in the same cluster
    
    clustering = DBSCAN(eps=10, min_samples=2).fit(X_coords)
    labels = clustering.labels_

    # Analyze Clusters
    for cluster_id in set(labels):
        if cluster_id == -1: continue # Skip noise
        
        cluster_mask = (labels == cluster_id)
        cluster_points = X_coords[cluster_mask]
        cluster_times = np.array([n['time'] for n in final_nulls])[cluster_mask]
        
        centroid = np.mean(cluster_points, axis=0)
        t_start, t_end = np.min(cluster_times), np.max(cluster_times)
        
        print(f"\n--- Cluster {cluster_id} ---")
        print(f"Detections: {len(cluster_points)}")
        print(f"Time Range: {t_start:.3f}s - {t_end:.3f}s")
        print(f"Centroid GSE: X={centroid[0]:.2f}, Y={centroid[1]:.2f}, Z={centroid[2]:.2f}")
end = time.time()

print("Runtime= ", end -start)

idx_min = 23926
sat_pos = np.array([R_masked[0][idx_min], R_masked[1][idx_min], R_masked[2][idx_min] ,R_masked[3][idx_min]])
print (sat_pos)
