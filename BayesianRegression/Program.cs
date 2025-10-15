using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace BayesianRegression;

class Program
{
    static void Main(string[] args)
    {
        //
        // Challenger O-ring data
        // Taken from: http://archive.ics.uci.edu/ml/machine-learning-databases/space-shuttle/o-ring-erosion-only.data
        //
        // This dataset contains information from 23 Space Shuttle flights before the Challenger disaster.
        // Each data point represents a flight with:
        // - temp: The temperature (°F) at launch
        // - distress: The number of O-rings showing thermal distress
        //

        double[] temp = { 66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58 };
        double[] distress = { 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

        Console.WriteLine("=== Bayesian Linear Regression for Challenger O-Ring Data ===\n");
        Console.WriteLine($"Training data: {temp.Length} flights");
        Console.WriteLine($"Temperature range: {temp.Min()}°F - {temp.Max()}°F\n");

        // put features into array of Vectors
        Vector[] xdata = new Vector[temp.Length];
        for (int i = 0; i < temp.Length; i++)
            xdata[i] = Vector.FromArray(temp[i], 1);    // including bias term

        //
        // Model variables
        //

        // define a prior distribution and attach that to "w" random variable
        // w[0] is the slope (temperature coefficient), w[1] is the intercept (bias)
        VectorGaussian wPrior = new VectorGaussian(Vector.Zero(2), PositiveDefiniteMatrix.Identity(2));
        Variable<Vector> w = Variable.Random(wPrior);

        // define prior for observation noise variance
        Gamma noiseDist = new Gamma(1, 2);
        Variable<double> noise = Variable.Random(noiseDist);

        // set features "x" and observations "y" as observed in the model
        VariableArray<double> y = Variable.Observed(distress);
        Range n = y.Range;
        VariableArray<Vector> x = Variable.Observed(xdata, n);

        // define "y" statistically: Gaussian RV array. 
        // Mean is defined by dot-product of param vector "w" and the feature vector x[n]
        y[n] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[n]), noise);

        //
        // Training: parameter inference
        //

        InferenceEngine engine = new InferenceEngine();
        engine.Compiler.RecommendedQuality = Microsoft.ML.Probabilistic.Factors.Attributes.QualityBand.Experimental;

        Console.WriteLine("Running Bayesian inference...\n");

        // infer "w" posterior as a distribution
        VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
        Gamma noisePosterior = engine.Infer<Gamma>(noise);
        
        Console.WriteLine("=== Inference Results ===");
        Console.WriteLine($"Weight vector posterior:\n{wPosterior}");
        Console.WriteLine($"\nNoise variance posterior:\n{noisePosterior}");

        // Extract mean values for interpretation
        Vector wMean = wPosterior.GetMean();
        Console.WriteLine($"\nModel equation: distress = {wMean[0]:F4} * temperature + {wMean[1]:F4}");
        Console.WriteLine($"(Negative slope indicates: lower temperature → more distress)\n");

        //
        // Prediction: temp = 31°F (actual temperature at Challenger launch)
        //

        double tempTest = 31;
        Console.WriteLine($"\n=== Prediction for Challenger Launch Temperature ({tempTest}°F) ===");

        // Create feature vector for test point
        Vector xdataTest = Vector.FromArray(tempTest, 1);
        
        // Use the posterior mean and variance for noise
        double noiseMean = noisePosterior.GetMean();
        
        // Calculate prediction using the posterior distributions
        // Mean prediction: w^T * x
        double meanPrediction = wMean[0] * tempTest + wMean[1];
        
        // Variance prediction: x^T * Cov(w) * x + noise_variance
        PositiveDefiniteMatrix wCovariance = wPosterior.GetVariance();
        double predictionVariance = xdataTest.Inner(wCovariance * xdataTest) + noiseMean;
        
        Gaussian distressPrediction = new Gaussian(meanPrediction, predictionVariance);
        
        Console.WriteLine($"Predicted distress distribution: {distressPrediction}");
        Console.WriteLine($"Mean predicted distress: {distressPrediction.GetMean():F2} O-rings");
        Console.WriteLine($"95% confidence interval: [{distressPrediction.GetMean() - 1.96 * Math.Sqrt(distressPrediction.GetVariance()):F2}, " +
                         $"{distressPrediction.GetMean() + 1.96 * Math.Sqrt(distressPrediction.GetVariance()):F2}]");
        Console.WriteLine($"\nNote: The model predicts that at {tempTest}°F (the actual Challenger launch temperature),");
        Console.WriteLine($"there would be significant O-ring distress, which tragically matches what occurred.");
    }
}
