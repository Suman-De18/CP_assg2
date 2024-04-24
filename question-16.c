#include <stdio.h>
#include <math.h>

double f(double t, double y) {
    return y - t * t + 1;
}

double exact_solution(double t) {
    return pow(t + 1, 2) - 0.5 * exp(t);
}

int main() {
    double t0 = 0.0;
    double y0 = 0.5;
    double L = 1.0;
    double M = 0.5 * exp(2) - 2;
    double h = 0.2;
    double t, y_euler, y_exact, error, error_bound;

    printf("t\t Euler's y(t)\t Exact y(t)\t Error\t\t Error Bound\n");
    printf("------------------------------------------------------------\n");

    for (t = t0+h; t <= 2.0; t += h) {
        // Exact solution
        y_exact = exact_solution(t);

        // Euler's method
        y_euler = y0 + h * f(t-h, y0);

        // Error
        error = fabs(y_exact - y_euler);

        // Error bound
        error_bound = (h * M) / (2 * L) * (exp(L * (t - 0)) - 1);

        printf("%.2f\t %.6f\t %.6f\t %.6f\t %.6f\n", t, y_euler, y_exact, error, error_bound);

        // Update y0 for the next iteration
        y0 = y_euler;
    }

    return 0;
}
