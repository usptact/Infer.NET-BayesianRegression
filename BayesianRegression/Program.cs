using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;


namespace BayesianRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            //
            // Challenger O-ring data
            // Taken from: http://archive.ics.uci.edu/ml/machine-learning-databases/space-shuttle/o-ring-erosion-only.data
            //

            double[] temp = { 66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58 };
            double[] distress = { 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

            // put features into array of Vectors
            Vector[] xdata = new Vector[temp.Length];
            for (int i = 0; i < temp.Length; i++)
                xdata[i] = Vector.FromArray(temp[i], 1);    // including bias

            //
            // Model variables
            //

            // define a prior distribution and attach that to "w" random variable
            VectorGaussian wPrior = new VectorGaussian(Vector.Zero(2), PositiveDefiniteMatrix.Identity(2));
            Variable<Vector> w = Variable.Random(wPrior);

            // hard-code variance
            Gamma noiseDist = new Gamma(1, 2);
            Variable<double> noise = Variable.Random(noiseDist);
            //double noise = 0.1;

            // set features "x" and observations "y" as observed in the model
            VariableArray<double> y = Variable.Observed(distress);
            Range n = y.Range;
            VariableArray<Vector> x = Variable.Observed(xdata, n);

            // define "y" statistically: Gaussian RV array. Mean is defined by dot-product of param vector "w" and the feature vector x[n]
            y[n] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[n]), noise);

            //
            // Training: parameter inference
            //

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = Microsoft.ML.Probabilistic.Factors.Attributes.QualityBand.Experimental;

            // infer "w" posterior as a distribution
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
            Gamma noisePosterior = engine.Infer<Gamma>(noise);
            Console.WriteLine("Distribution over w = \n" + wPosterior);
            Console.WriteLine("Distribution over noise = \n" + noisePosterior);

            //
            // Prediction: temp = 31
            //

            // one data point
            double tempTest = 31;

            // RV for prediction
            Variable<double> distressTest = Variable.Observed(tempTest);

            // RV for feature vector
            Vector xdataTest = Vector.FromArray(tempTest, 1);
            Variable<Vector> xTest = Variable.Observed(xdataTest);

            // set w distribution that was obtained from training
            Variable<Vector> wParam = Variable.Random(wPosterior);

            Variable<double> noiseParam = Variable.Random(noisePosterior);

            // RV for prediction
            distressTest = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(wParam, xTest), noiseParam);

            // infer and print prediction distribution
            Console.WriteLine("Test distress = \n" + engine.Infer(distressTest));

            Console.WriteLine("Press any key ...");
            Console.ReadKey();
        }
    }
}
