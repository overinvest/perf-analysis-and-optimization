#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

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

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t iN = i * n;
        for (std::size_t j = 0; j < n; ++j) {
            bt[j * n + i] = b[iN + j];
        }
    }
}

static void matmul_tiled_transposedB(const Matrix& A, const Matrix& Bt, Matrix& C, std::size_t BS) {
    const std::size_t n = A.n;
    const double* a = A.d.data();
    const double* bt = Bt.d.data();
    double* c = C.d.data();

    std::fill(c, c + n * n, 0.0);

    for (std::size_t ii = 0; ii < n; ii += BS) {
        const std::size_t iimax = std::min(ii + BS, n);
        for (std::size_t jj = 0; jj < n; jj += BS) {
            const std::size_t jjmax = std::min(jj + BS, n);
            for (std::size_t kk = 0; kk < n; kk += BS) {
                const std::size_t kkmax = std::min(kk + BS, n);

                for (std::size_t i = ii; i < iimax; ++i) {
                    const std::size_t iN = i * n;

                    for (std::size_t j = jj; j < jjmax; ++j) {
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
    matmul_tiled_transposedB(A, Bt, C, BS);
    const auto t1 = std::chrono::steady_clock::now();

    volatile double sink = checksum(C);
    (void)sink;

    std::chrono::duration<double> dt = t1 - t0;
    return dt.count();
}

int main(int argc, char** argv) {
    std::size_t n = 768;
    int repeats = 7;
    std::size_t BS = 64; // 32

    if (argc >= 2) n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (argc >= 3) repeats = std::stoi(argv[2]);
    if (argc >= 4) BS = static_cast<std::size_t>(std::stoul(argv[3]));
    if (repeats < 1) repeats = 1;
    if (BS == 0) BS = 64;

    Matrix A(n), B(n), Bt(n), C(n);

    (void)run_once(A, B, Bt, C, BS); // прогрев

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeats));

    for (int r = 0; r < repeats; ++r) {
        times.push_back(run_once(A, B, Bt, C, BS));
    }

    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    const double best = times.front();
    const double worst = times.back();

    std::cout << "N=" << n
              << " BS=" << BS
              << " repeats=" << repeats
              << " median=" << std::fixed << std::setprecision(6) << median << " s"
              << " best=" << best << " s"
              << " worst=" << worst << " s\n";
    return 0;
}
