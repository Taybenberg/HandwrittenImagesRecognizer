using System.Collections.Generic;

namespace HandwrittenNumbersRecognition
{
    public class NeuralNet
    {
        public double[] BiasWeights { get; set; }

        public double[][][] Weights { get; set; }

        List<double[]> layers = new List<double[]>();

        public NeuralNet()
        {
            Weights = new double[Settings.hiddenLayersSizes.Length + 1][][];

            BiasWeights = new double[Settings.hiddenLayersSizes.Length + 1];

            for (int i = 0; i < Settings.hiddenLayersSizes.Length + 1; ++i)
            {
                BiasWeights[i] = Settings.random();

                Weights[i] = new double[Settings.getLayerSize(i + 1)][];

                for (int j = 0; j < Settings.getLayerSize(i + 1); ++j)
                {
                    Weights[i][j] = new double[Settings.getLayerSize(i)];

                    for (int z = 0; z < Settings.getLayerSize(i); ++z)
                        Weights[i][j][z] = Settings.random();
                } 
            }
        }

        ~NeuralNet()
        {
            for (int i = 0; i < Settings.hiddenLayersSizes.Length + 1; ++i)
            {
                for (int j = 0; j < Settings.getLayerSize(i + 1); ++j)
                    Weights[i][j] = null;

                Weights[i] = null;
            }

            Weights = null;

            BiasWeights = null;
        }

        public double[] Output(byte[] input)
        {
            layers.Clear();
            
            layers.Add(new double[Settings.getLayerSize(0)]);

            double[] tmpLayer;

            input.CopyTo(layers[^1], 0);

            for (int i = 0; i < Settings.hiddenLayersSizes.Length + 1; ++i)
            {
                tmpLayer = new double[Settings.getLayerSize(i + 1)];

                for (int j = 0; j < Settings.getLayerSize(i + 1); ++j)
                {
                    double sum = Settings.bias * BiasWeights[i];

                    for (int z = 0; z < Settings.getLayerSize(i); ++z)
                        sum += layers[^1][z] * Weights[i][j][z];

                    tmpLayer[j] = Settings.activationFunction(sum);
                }

                layers.Add(new double[Settings.getLayerSize(i + 1)]);

                tmpLayer.CopyTo(layers[^1], 0);
            }

            return layers[^1];
        }

        public void BackPropagation(double[] correctOutput)
        {
            double[] tmpLayer = new double[correctOutput.Length];

            double errorSum = 0.0;

            for (int i = 0; i < correctOutput.Length; ++i)
            {
                tmpLayer[i] = (layers[^1][i] - correctOutput[i]) * (layers[^1][i] * (1.0 - layers[^1][i]));
                errorSum += tmpLayer[i] * Settings.bias;
            }

            layers.RemoveAt(layers.Count - 1);

            for (int i = Settings.hiddenLayersSizes.Length; i >= 0; --i)
            {
                BiasWeights[i] -= Settings.learningSpeed * errorSum;

                double[] hiddenLayer = new double[layers[^1].Length];

                for (int j = 0; j < Settings.getLayerSize(i + 1); ++j)
                {
                    for (int z = 0; z < Settings.getLayerSize(i); ++z)
                    {
                        hiddenLayer[z] += tmpLayer[j] * Weights[i][j][z];
                        Weights[i][j][z] -= Settings.learningSpeed * tmpLayer[j] * layers[^1][z];
                    }
                }

                if (i > 0)
                {
                    tmpLayer = new double[hiddenLayer.Length];

                    errorSum = 0.0;

                    for (int j = 0; j < tmpLayer.Length; ++j)
                    {
                        tmpLayer[j] = hiddenLayer[j] * (layers[^1][j] * (1.0 - layers[^1][j]));
                        errorSum += tmpLayer[j] * Settings.bias;
                    }

                    layers.RemoveAt(layers.Count - 1);
                }
            }
        }
    }
}