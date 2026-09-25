"""Majorana SYK model: spectrum of the q=4 SYK Hamiltonian.

GPU-accelerated. The Hamiltonian is built from the Pauli-string (signed
permutation) representation of the Majorana operators rather than by dense
matrix multiplication, and the diagonalizations are batched onto the GPU.
Falls back to NumPy/CPU automatically when CuPy is unavailable.
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- backend

try:
    import cupy as _cp
    import cupyx as _cpx

    _cp.cuda.runtime.getDeviceCount()  # raises if no usable device
    xp = _cp
    GPU = True
except Exception:  # no cupy, no driver, no device
    _cpx = None
    xp = np
    GPU = False


def backend_name():
    if not GPU:
        return "NumPy (CPU)"
    props = xp.cuda.runtime.getDeviceProperties(xp.cuda.runtime.getDevice())
    return f"CuPy (GPU: {props['name'].decode()})"


def to_cpu(a):  # bring a device array back to NumPy
    return xp.asnumpy(a) if GPU else np.asarray(a)


def _scatter_add(target, flat_indices, values):
    """target.flat[flat_indices] += values, accumulating duplicates.

    target must be a contiguous real array: cupyx.scatter_add has no complex
    kernel, so complex data is accumulated as two real parts by the caller.
    (cupy.scatter_add itself was removed from the top-level namespace.)
    """
    if GPU:
        _cpx.scatter_add(target.reshape(-1), flat_indices.reshape(-1),
                         values.reshape(-1))
    else:
        np.add.at(target.reshape(-1), flat_indices.reshape(-1),
                  values.reshape(-1))


def _parity(v):
    """popcount(v) mod 2, for unsigned 64-bit integer arrays."""
    v = v ^ (v >> xp.uint64(32))
    v = v ^ (v >> xp.uint64(16))
    v = v ^ (v >> xp.uint64(8))
    v = v ^ (v >> xp.uint64(4))
    v = v ^ (v >> xp.uint64(2))
    v = v ^ (v >> xp.uint64(1))
    return (v & xp.uint64(1)).astype(xp.int8)


# ------------------------------------------------------- dense references
# Kept for validating the fast path at small N.  Never call these with a
# large N: each matrix is (2**(N//2))**2 complex entries.


def pauli_matrices():  # define the Pauli matrices
    paulis = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }
    return paulis


def kron_list(matrices_list):  # tensor product of matrices

    input_matrix = matrices_list[0]

    for matrix in matrices_list[1:]:
        input_matrix = np.kron(input_matrix, matrix)

    return input_matrix


def test_pauli_matrices():  # testing Pauli matrices
    p = pauli_matrices()
    I, X, Y, Z = p['I'], p['X'], p['Y'], p['Z']

    for name, M in p.items():
        assert np.allclose(
            M, M.conj().T
        ), f"{name} is not Hermitian"

    assert np.allclose(X @ X, I)
    assert np.allclose(Y @ Y, I)
    assert np.allclose(Z @ Z, I)

    assert np.allclose(
        X @ Y + Y @ X,
        np.zeros((2, 2))
    )

    assert np.allclose(
        X @ Z + Z @ X,
        np.zeros((2, 2))
    )

    assert np.allclose(
        Y @ Z + Z @ Y,
        np.zeros((2, 2))
    )

    assert np.allclose(X @ Y, 1j * Z)

    print("pauli_matrices tests passed")


def build_majoranas(N):  # dense reference construction

    M = N // 2  # each two Majoranas need one qubit

    p = pauli_matrices()

    majoranas = []

    for K in range(M):

        chi_2K = (1 / np.sqrt(2)) * kron_list(
            [
                p['Z'] if i < K else p['X'] if i == K else p['I'] for i in range(M)]
        )

        chi_2K_plus_1 = (1 / np.sqrt(2)) * kron_list(
            [
                p['Z'] if i < K else p['Y'] if i == K else p['I'] for i in range(M)]
        )

        majoranas.append(chi_2K)
        majoranas.append(chi_2K_plus_1)

    return majoranas


# ------------------------------------------------- Pauli-string Majoranas
# chi_a = coef_a * X^(x_a) Z^(z_a), a signed permutation matrix: column n
# carries a single entry, at row n ^ x_a, with value
# coef_a * (-1)**popcount(z_a & n).  Qubit k occupies bit (M-1-k) of n, which
# matches the ordering produced by kron_list.


def majorana_strings(N):
    """Return (x, z, coef) arrays of length N describing the Majoranas."""

    M = N // 2

    x = np.zeros(N, dtype=np.uint64)
    z = np.zeros(N, dtype=np.uint64)
    coef = np.zeros(N, dtype=np.complex128)

    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    for K in range(M):
        bit_K = np.uint64(1) << np.uint64(M - 1 - K)
        z_tail = np.uint64(0)
        for i in range(K):  # Z on every qubit before K
            z_tail |= np.uint64(1) << np.uint64(M - 1 - i)

        # chi_{2K} = Z...Z X I...I
        x[2 * K] = bit_K
        z[2 * K] = z_tail
        coef[2 * K] = inv_sqrt2

        # chi_{2K+1} = Z...Z Y I...I,  with Y = i * X * Z
        x[2 * K + 1] = bit_K
        z[2 * K + 1] = z_tail | bit_K
        coef[2 * K + 1] = 1j * inv_sqrt2

    return x, z, coef


def quadruple_strings(N):
    """Pauli string of chi_i chi_j chi_k chi_l for every i<j<k<l.

    Returns (x, z, coef) device arrays of length C(N, 4).
    """

    x1, z1, c1 = majorana_strings(N)

    idx = np.array(list(itertools.combinations(range(N), 4)), dtype=np.int64)
    idx = xp.asarray(idx)
    x1, z1, c1 = xp.asarray(x1), xp.asarray(z1), xp.asarray(c1)

    acc_x = xp.zeros(idx.shape[0], dtype=xp.uint64)
    acc_z = xp.zeros(idx.shape[0], dtype=xp.uint64)
    acc_c = xp.ones(idx.shape[0], dtype=xp.complex128)

    for column in range(4):
        a = idx[:, column]
        # (X^acc_x Z^acc_z)(X^x Z^z) = (-1)^popcount(acc_z & x) X^(..) Z^(..)
        sign = 1.0 - 2.0 * _parity(acc_z & x1[a]).astype(xp.float64)
        acc_c = acc_c * c1[a] * sign
        acc_x = acc_x ^ x1[a]
        acc_z = acc_z ^ z1[a]

    return acc_x, acc_z, acc_c


def build_hamiltonian(N, J=1.0, rng=None, term_chunk=512):
    """Dense SYK Hamiltonian on the active backend, dtype complex128."""

    rng = np.random.default_rng() if rng is None else rng

    M = N // 2
    dim = 2 ** M

    sigma = np.sqrt(6 * J**2 / N**3)

    x, z, coef = quadruple_strings(N)
    n_terms = x.shape[0]

    J_ijkl = xp.asarray(rng.normal(scale=sigma, size=n_terms))
    coef = coef * J_ijkl

    # Terms sharing an x mask hit the same matrix positions: sum them into one
    # diagonal vector per distinct mask, then scatter once.
    ux, inverse = xp.unique(x, return_inverse=True)
    inverse = inverse.reshape(-1).astype(xp.int64)

    n_cols = xp.arange(dim, dtype=xp.uint64)
    cols = xp.arange(dim, dtype=xp.int64)
    diag_re = xp.zeros((ux.shape[0], dim), dtype=xp.float64)
    diag_im = xp.zeros((ux.shape[0], dim), dtype=xp.float64)

    for start in range(0, n_terms, term_chunk):
        stop = min(start + term_chunk, n_terms)
        signs = 1.0 - 2.0 * _parity(
            z[start:stop, None] & n_cols[None, :]
        ).astype(xp.float64)
        block = coef[start:stop, None] * signs

        flat = inverse[start:stop, None] * dim + cols[None, :]
        _scatter_add(diag_re, flat, xp.ascontiguousarray(block.real))
        _scatter_add(diag_im, flat, xp.ascontiguousarray(block.imag))

    diagonals = diag_re + 1j * diag_im
    del diag_re, diag_im

    # Place each diagonal: column n of mask group g lands on row n ^ ux[g].
    # Distinct masks never collide on a position, so this is assignment, not
    # accumulation -- which keeps it off the scatter path entirely.
    H = xp.zeros((dim, dim), dtype=xp.complex128)
    for g in range(0, ux.shape[0], term_chunk):
        gstop = min(g + term_chunk, ux.shape[0])
        target_rows = (n_cols[None, :] ^ ux[g:gstop, None]).astype(xp.int64)
        H[target_rows, cols[None, :]] = diagonals[g:gstop]

    return H


def build_hamiltonian_dense_reference(N, J=1.0, rng=None):
    """Original O(dim**3) construction -- small N only, for validation."""

    rng = np.random.default_rng() if rng is None else rng

    majoranas = build_majoranas(N)
    dim = 2 ** (N // 2)
    sigma = np.sqrt(6 * J**2 / N**3)

    H = np.zeros((dim, dim), dtype=complex)

    quads = list(itertools.combinations(range(N), 4))
    couplings = rng.normal(scale=sigma, size=len(quads))

    for t, (i, j, k, l) in enumerate(quads):
        term = majoranas[i] @ majoranas[j] @ majoranas[k] @ majoranas[l]
        H += couplings[t] * term

    return H


# -------------------------------------------------------------- diagnostics


def test_majoranas(N):
    """Clifford algebra check, done on the Pauli strings (no dim**3 work)."""

    x, z, coef = majorana_strings(N)
    dim = 2 ** (N // 2)

    assert len(x) == N, f"Expected {N} majoranas, got {len(x)}"

    for a in range(N):
        # chi is Hermitian iff coef * (-1)**popcount(x & z) == conj(coef)
        phase = -1.0 if bin(int(x[a] & z[a])).count("1") % 2 else 1.0
        assert np.isclose(coef[a] * phase, np.conj(coef[a])), \
            f"chi_{a} is not Hermitian"

    for i in range(N):
        for j in range(i + 1, N):
            # chi_i chi_j = (-1)**(popcount(z_i & x_j) + popcount(x_i & z_j))
            #               chi_j chi_i; they anticommute iff that sign is -1
            p = (bin(int(z[i] & x[j])).count("1")
                 + bin(int(x[i] & z[j])).count("1"))
            assert p % 2 == 1, f"anticommutator(chi_{i}, chi_{j}) != 0"

    print(f"All Clifford algebra tests passed for N={N}, dimension={dim}")


def test_fast_matches_dense(N=8, J=1.0, seed=0):
    """The fast builder must reproduce the dense one term for term."""

    H_fast = to_cpu(build_hamiltonian(N, J=J, rng=np.random.default_rng(seed)))
    H_ref = build_hamiltonian_dense_reference(
        N, J=J, rng=np.random.default_rng(seed)
    )

    assert np.allclose(H_fast, H_ref), \
        "fast Hamiltonian disagrees with the dense reference"

    print(f"fast/dense Hamiltonian agreement verified for N={N}")


def test_hamiltonian(N, rng=None):

    H = build_hamiltonian(N, J=1.0, rng=rng)

    assert xp.allclose(H, H.conj().T), "Hamiltonian is not Hermitian!"

    dim = 2 ** (N // 2)

    assert H.shape == (dim, dim)

    print(f"Hamiltonian tests passed for N={N}, dimension={dim}")

    return H


# ---------------------------------------------------------- diagonalization


def diagonalize(H):

    return to_cpu(xp.linalg.eigvalsh(xp.asarray(H)))


def _auto_batch(dim, requested=None):
    """How many Hamiltonians to diagonalize in one batched call."""

    if requested is not None:
        return max(1, requested)
    if not GPU:
        return 1

    free, _total = xp.cuda.runtime.memGetInfo()
    bytes_per_H = dim * dim * 16
    # The batched solver (syevj) always computes eigenvectors, so it needs a
    # second B x dim x dim array plus workspace on top of the batch itself:
    # budget ~3x, and leave headroom on the card.
    return max(1, min(32, int(0.70 * free / (3 * bytes_per_H))))


def _oom_error():
    return xp.cuda.memory.OutOfMemoryError if GPU else ()


def _free_pool():
    if GPU:
        xp.get_default_memory_pool().free_all_blocks()


def _diagonalize_batch(N, n, J, rng, dim):
    """Build and diagonalize n Hamiltonians in one batched solver call."""

    stack = xp.empty((n, dim, dim), dtype=xp.complex128)
    try:
        for b in range(n):
            stack[b] = build_hamiltonian(N, J=J, rng=rng)
        return to_cpu(xp.linalg.eigvalsh(stack))  # batched on device
    finally:
        del stack
        _free_pool()


def run_multiple_realizations(N, n_realizations, J=1.0, batch_size=None,
                              seed=None, verbose=True):
    """Diagonalize n_realizations SYK Hamiltonians, batched on the GPU.

    The batch size is chosen from free device memory and halved automatically
    if the solver still runs out.
    """

    rng = np.random.default_rng(seed)

    dim = 2 ** (N // 2)
    batch = _auto_batch(dim, batch_size)

    if verbose:
        print(f"diagonalizing {n_realizations} realizations of dim {dim} "
              f"in batches of {batch} on {backend_name()}")

    all_eigenvalues = np.empty(n_realizations * dim, dtype=np.float64)
    done = 0

    while done < n_realizations:
        this_batch = min(batch, n_realizations - done)

        try:
            eigvals = _diagonalize_batch(N, this_batch, J, rng, dim)
        except _oom_error():
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            if verbose:
                print(f"\nout of device memory; retrying with batch {batch}")
            _free_pool()
            continue  # same realizations, smaller batch

        all_eigenvalues[done * dim:(done + this_batch) * dim] = eigvals.ravel()
        done += this_batch

        if verbose:
            print(f"  {done}/{n_realizations}", end="\r", flush=True)

    if verbose:
        print()

    return all_eigenvalues


def plot_energy_histogram(all_eigenvalues, N, n_realizations, bins=60):

    plt.figure(figsize=(7, 5))

    plt.hist(all_eigenvalues, bins=bins, density=True, edgecolor='black',
             alpha=0.7)

    plt.xlabel("Energy E")
    plt.ylabel("Density of states")

    plt.title(f"SYK energy spectrum, N={N}, {n_realizations} realizations")

    plt.savefig(f"syk_histogram_N{N}.png", dpi=150)

    plt.show()

    print(f"Histogram saved as syk_histogram_N{N}.png")


if __name__ == "__main__":

    print(f"backend: {backend_name()}")

    test_pauli_matrices()
    test_fast_matches_dense(N=8)

    N = 16
    n_realizations = 200

    test_majoranas(N)

    H_test = test_hamiltonian(N)

    eigs_test = diagonalize(H_test)

    print(f"Eigenvalues (N={N}):")
    print(eigs_test)

    del H_test
    if GPU:
        xp.get_default_memory_pool().free_all_blocks()

    eigs_N0 = run_multiple_realizations(N, n_realizations=n_realizations, J=1.0)

    plot_energy_histogram(eigs_N0, N, n_realizations=n_realizations)
