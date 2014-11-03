/*
 * Maximum Likelihood estimator using Google's ceres-solver
 *
 */

#include <iomanip>
#include <fstream>
#include <iterator>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

using namespace boost::accumulators;

// Log likelihood of normal distribution
double llfunc(double x, double mu, double sigma) {
    return -log(1 / (sigma * sqrt(2 * M_PI))) +
           (x - mu) * (x - mu) / (2 * sigma * sigma);
}

// Manual derivative with respect to mu
double llfunc_div_mu(double x, double mu, double sigma) {
    return (mu - x) / (sigma * sigma);
}

// Manual derivative with respect to sigma
double llfunc_div_sigma(double x, double mu, double sigma) {
    return -(mu * mu - 2 * mu * x - sigma * sigma + x * x) /
           (sigma * sigma * sigma);
}

/*
 * Class representing the log likelihood for a certain dataset
 * OpenMP is used to process parts of the dataset in parallel
 */
class LogLikelihood : public ceres::FirstOrderFunction {
public:
    LogLikelihood(std::vector<double> input) : data(input) {}

    virtual ~LogLikelihood() {}

    virtual bool Evaluate(const double* parameters, double* cost,
                          double* gradient) const {
        const double mu = parameters[0];
        const double sigma = parameters[1];

        if (mu < -10 || mu > 10 || sigma <= 0) {
            return false;
        }

        // Calculate cost
        double c = 0;
        #pragma omp parallel for reduction(+ : c)
        for (size_t i = 0; i < data.size(); i++) {
            c += llfunc(data[i], mu, sigma);
        }
        cost[0] = c;

        // Calculate gradient
        if (gradient != NULL) {
            double g0 = 0;
            double g1 = 0;
            #pragma omp parallel for reduction(+ : g0, g1)
            for (size_t i = 0; i < data.size(); i++) {
                g0 += llfunc_div_mu(data[i], mu, sigma);
                g1 += llfunc_div_sigma(data[i], mu, sigma);
            }
            gradient[0] = g0;
            gradient[1] = g1;
        }
        return true;
    }

    virtual int NumParameters() const { return 2; }

private:
    std::vector<double> data;
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    double parameters[2] = {8, 8.0};

    // Generate data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, 1);
    std::vector<double> input;
    std::cout << "Generating data..." << std::endl;
    for (int i = 0; i < 100000; i++) {
        input.push_back(dis(gen));
    }

    // Perform fit
    std::cout << "Running fit..." << std::endl;
    ceres::GradientProblemSolver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 300;
    options.function_tolerance = 1e-15;

    ceres::GradientProblemSolver::Summary summary;
    ceres::GradientProblem problem(new LogLikelihood(input));
    ceres::Solve(options, problem, parameters, &summary);

    //std::cout << summary.FullReport() << std::endl;

    std::cout << std::setprecision(16)
              << "Final    mu: " << parameters[0] << " sigma: " << parameters[1]
              << std::endl;

    // Compare with analytic estimators (mean and std)
    accumulator_set<double, features<tag::mean, tag::variance>> acc;
    for_each(input.begin(), input.end(), [&acc](double x) { acc(x); });

    std::cout << std::setprecision(16)
              << "Analytic mu: " << mean(acc) << " sigma: " << sqrt(variance(acc))
              << std::endl;

    std::ofstream data("./data.txt");
    std::ostream_iterator<double> output_iterator(data, "\n");
    std::copy(input.begin(), input.end(), output_iterator);

    std::ofstream params("./params.txt");
    params << parameters[0] << std::endl;
    params << parameters[1] << std::endl;

    return 0;
}

