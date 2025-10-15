# Infer.NET Bayesian Regression - Challenger O-Ring Analysis

A demonstration of Bayesian linear regression using Microsoft's Infer.NET framework, applied to the historical Challenger Space Shuttle O-ring failure data.

## Overview

This project implements a Bayesian linear regression model to analyze the relationship between launch temperature and O-ring thermal distress in Space Shuttle flights. The analysis uses data from 23 flights prior to the Challenger disaster to predict O-ring failures at low temperatures.

## The Model

### Statistical Model

The Bayesian linear regression model is defined as:

```
y_i ~ Gaussian(w^T * x_i, σ²)
```

Where:
- `y_i` is the number of O-rings showing thermal distress
- `x_i` is the feature vector [temperature, 1] (includes bias term)
- `w` is the weight vector [slope, intercept]
- `σ²` is the observation noise variance

### Priors

- **Weight vector prior**: `w ~ VectorGaussian(0, I)` - uninformative prior
- **Noise variance prior**: `σ² ~ Gamma(1, 2)` - weak prior on observation noise

### Inference

The model uses **Expectation Propagation (EP)** algorithm to compute the posterior distributions:
- `p(w | data)` - posterior distribution over weight parameters
- `p(σ² | data)` - posterior distribution over noise variance

### Prediction

For a new temperature value, predictions are made using:

```
p(y_new | data) = ∫∫ p(y_new | w, σ²) p(w | data) p(σ² | data) dw dσ²
```

The prediction mean is `w_mean^T * x_new` and the variance accounts for both parameter uncertainty and observation noise.

## Dataset

The model uses historical data from 23 Space Shuttle flights:
- **Temperature**: Launch temperature in degrees Fahrenheit (53°F to 81°F)
- **O-ring distress**: Count of O-rings showing thermal distress (0 to 2)

Source: [UCI Machine Learning Repository - Space Shuttle O-ring Data](http://archive.ics.uci.edu/ml/machine-learning-databases/space-shuttle/o-ring-erosion-only.data)

## Results

The model learns that:
1. There is a **negative correlation** between temperature and O-ring distress
2. Lower temperatures lead to higher predicted distress
3. At 31°F (the actual Challenger launch temperature), the model predicts significant O-ring distress (mean: ~1.34 O-rings)

This analysis demonstrates how Bayesian methods can quantify uncertainty and provide probabilistic predictions that could have informed the tragic decision on January 28, 1986.

## Requirements

- **.NET 8.0 SDK** or later
- **Infer.NET 0.4.2504.701** (automatically installed via NuGet)

## Building the Project

### Using .NET CLI

```bash
# Navigate to the project directory
cd BayesianRegression

# Restore dependencies
dotnet restore

# Build the project
dotnet build

# Run the application
dotnet run
```

### Using Visual Studio

1. Open `BayesianRegression.sln` in Visual Studio 2022 or later
2. Build the solution (Ctrl+Shift+B or Build > Build Solution)
3. Run the project (F5 or Debug > Start Debugging)

### Using Visual Studio Code

1. Open the project folder in VS Code
2. Install the C# Dev Kit extension
3. Press F5 to build and run

## Project Structure

```
Infer.NET-BayesianRegression/
├── BayesianRegression/
│   ├── BayesianRegression.csproj    # Modern SDK-style project file
│   └── Program.cs                    # Main application code
├── BayesianRegression.sln            # Solution file
├── README.md                          # This file
└── LICENSE                            # Apache 2.0 License
```

## Example Output

```
=== Bayesian Linear Regression for Challenger O-Ring Data ===

Training data: 23 flights
Temperature range: 53°F - 81°F

Running Bayesian inference...

Compiling model...done.
Iterating: 
.........|.........|.........|.........|.........| 50
=== Inference Results ===
Weight vector posterior:
VectorGaussian(-0.02742 2.191, 0.0001102 -0.007655)
                               -0.007655 0.5416   

Noise variance posterior:
Gamma(12.11, 0.01916)[mean=0.232]

Model equation: distress = -0.0274 * temperature + 2.1907
(Negative slope indicates: lower temperature → more distress)


=== Prediction for Challenger Launch Temperature (31°F) ===
Predicted distress distribution: Gaussian(1.341, 0.4049)
Mean predicted distress: 1.34 O-rings
95% confidence interval: [0.09, 2.59]

Note: The model predicts that at 31°F (the actual Challenger launch temperature),
there would be significant O-ring distress, which tragically matches what occurred.
```

## Technical Details

### Framework

This project uses [Infer.NET](https://dotnet.github.io/infer/), Microsoft's open-source framework for running Bayesian inference in graphical models. Infer.NET automatically:
- Compiles the model into efficient inference code
- Chooses appropriate message-passing algorithms
- Handles numerical stability and convergence

### Key Classes Used

- `Variable<T>`: Represents random variables in the model
- `VectorGaussian`: Multivariate Gaussian distribution
- `Gamma`: Gamma distribution for positive real-valued variables
- `InferenceEngine`: Performs inference using message-passing algorithms

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. (Chapter 3: Linear Models for Regression)
2. Minka, T. et al. (2018). *Infer.NET*. Microsoft Research. https://dotnet.github.io/infer/
3. Dalal, S. R., Fowlkes, E. B., & Hoadley, B. (1989). "Risk Analysis of the Space Shuttle: Pre-Challenger Prediction of Failure." *Journal of the American Statistical Association*, 84(408), 945-957.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Infer.NET team at Microsoft Research for the probabilistic programming framework
- UCI Machine Learning Repository for providing the Challenger O-ring dataset
- The original statistical analysis by Dalal, Fowlkes, and Hoadley (1989)

## Contributing

This is a demonstration project. Feel free to fork and extend it with:
- Different priors or model structures
- Additional inference algorithms (Variational Bayes, Gibbs Sampling)
- Visualization of posterior distributions
- Cross-validation or model comparison

## Contact

For questions about Infer.NET, visit:
- Documentation: https://dotnet.github.io/infer/
- GitHub: https://github.com/dotnet/infer
- Issues: https://github.com/dotnet/infer/issues
