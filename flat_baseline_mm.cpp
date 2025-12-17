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

    inline double& at(std::size_t i, std::size_t j) {
        return d[i * n + j];
    }
    inline const double& at(std::size_t i, std::size_t j) const {
        return d[i * n + j];
    }
};

static void fill_random(Matrix& m, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    const std::size_t n = m.n;
    double* p = m.d.data();
    for (std::size_t i = 0; i < n * n; ++i) {
        p[i] = dist(rng);
    }
}

static void matmul_baseline_flat(const Matrix& A, const Matrix& B, Matrix& C) {
    const std::size_t n = A.n;
    const double* a = A.d.data();
    const double* b = B.d.data();
    double* c = C.d.data();

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t iN = i * n;
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += a[iN + k] * b[k * n + j];
            }
            c[iN + j] = sum;
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

static double run_once(Matrix& A, Matrix& B, Matrix& C) {
    fill_random(A, 42);
    fill_random(B, 1337);

    const auto t0 = std::chrono::steady_clock::now();
    matmul_baseline_flat(A, B, C);
    const auto t1 = std::chrono::steady_clock::now();

    volatile double sink = checksum(C);
    (void)sink;

    std::chrono::duration<double> dt = t1 - t0;
    return dt.count();
}

int main(int argc, char** argv) {
    std::size_t n = 768;
    int repeats = 7;

    if (argc >= 2) n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (argc >= 3) repeats = std::stoi(argv[2]);
    if (repeats < 1) repeats = 1;

    Matrix A(n), B(n), C(n);

    // Прогрев
    (void)run_once(A, B, C);

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeats));

    for (int r = 0; r < repeats; ++r) {
        times.push_back(run_once(A, B, C));
    }

    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    const double best = times.front();
    const double worst = times.back();

    std::cout << "N=" << n
              << " repeats=" << repeats
              << " median=" << std::fixed << std::setprecision(6) << median << " s"
              << " best=" << best << " s"
              << " worst=" << worst << " s\n";
    return 0;
}
