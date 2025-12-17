#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Matrix {
    std::size_t n{};
    std::vector<double> d;
    explicit Matrix(std::size_t n_) : n(n_), d(n_ * n_) {}
};

static void fill_random(Matrix& m, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    double* p = m.d.data();
    const std::size_t nn = m.n * m.n;
    for (std::size_t i = 0; i < nn; ++i) p[i] = dist(rng);
}

static void transpose(const Matrix& B, Matrix& Bt) {
    const std::size_t n = B.n;
    const double* b = B.d.data();
    double* bt = Bt.d.data();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        const std::size_t iN = static_cast<std::size_t>(i) * n;
        for (std::size_t j = 0; j < n; ++j) {
            bt[j * n + static_cast<std::size_t>(i)] = b[iN + j];
        }
    }
}

static void matmul_tiled_transposedB_omp(const Matrix& A, const Matrix& Bt, Matrix& C, std::size_t BS) {
    const std::size_t n = A.n;
    const double* a = A.d.data();
    const double* bt = Bt.d.data();
    double* c = C.d.data();

    std::fill(c, c + n * n, 0.0);

    const std::int64_t n64  = static_cast<std::int64_t>(n);
    const std::int64_t bs64 = static_cast<std::int64_t>(BS);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (std::int64_t ii = 0; ii < n64; ii += bs64) {
        for (std::int64_t jj = 0; jj < n64; jj += bs64) {

            const std::size_t i0 = static_cast<std::size_t>(ii);
            const std::size_t j0 = static_cast<std::size_t>(jj);
            const std::size_t iimax = std::min(i0 + BS, n);
            const std::size_t jjmax = std::min(j0 + BS, n);

            for (std::size_t kk = 0; kk < n; kk += BS) {
                const std::size_t kkmax = std::min(kk + BS, n);

                for (std::size_t i = i0; i < iimax; ++i) {
                    const std::size_t iN = i * n;
                    for (std::size_t j = j0; j < jjmax; ++j) {
                        const std::size_t jN = j * n;

                        double sum = c[iN + j];
                        for (std::size_t k = kk; k < kkmax; ++k) {
                            sum += a[iN + k] * bt[jN + k];
                        }
                        c[iN + j] = sum;
                    }
                }
            }
        }
    }
}

static double checksum(const Matrix& C) {
    const double* p = C.d.data();
    const std::size_t nn = C.n * C.n;
    double s = 0.0;
    for (std::size_t i = 0; i < nn; ++i) s += p[i];
    return s;
}

static double run_once(Matrix& A, Matrix& B, Matrix& Bt, Matrix& C, std::size_t BS) {
    fill_random(A, 42);
    fill_random(B, 1337);

    transpose(B, Bt);

    const auto t0 = std::chrono::steady_clock::now();
    matmul_tiled_transposedB_omp(A, Bt, C, BS);
    const auto t1 = std::chrono::steady_clock::now();

    volatile double sink = checksum(C);
    (void)sink;

    std::chrono::duration<double> dt = t1 - t0;
    return dt.count();
}

int main(int argc, char** argv) {
    std::size_t n = 768;
    int repeats = 7;
    std::size_t BS = 64;

    if (argc >= 2) n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (argc >= 3) repeats = std::stoi(argv[2]);
    if (argc >= 4) BS = static_cast<std::size_t>(std::stoul(argv[3]));
    if (repeats < 1) repeats = 1;
    if (BS == 0) BS = 64;

#ifdef _OPENMP
    if (argc >= 5) {
        int threads = std::stoi(argv[4]);
        if (threads > 0) omp_set_num_threads(threads);
    }
#endif

    Matrix A(n), B(n), Bt(n), C(n);

    (void)run_once(A, B, Bt, C, BS); // прогрев

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeats));

    for (int r = 0; r < repeats; ++r) {
        times.push_back(run_once(A, B, Bt, C, BS));
    }

    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];

#ifdef _OPENMP
    const int th = omp_get_max_threads();
#else
    const int th = 1;
#endif

    std::cout << "N=" << n
              << " BS=" << BS
              << " threads=" << th
              << " repeats=" << repeats
              << " median=" << std::fixed << std::setprecision(6) << median << " s\n";
    return 0;
}
