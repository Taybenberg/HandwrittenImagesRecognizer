using System;

namespace HandwrittenNumbersRecognition
{
    public static class Settings
    {
        public static Random R = new Random();

        public static double bias = 1.0;

        public static double learningSpeed = 0.9;

        public static int imageHeight = 28;

        public static int imageWidth = 28;

        public static int inputs = imageHeight * imageWidth;

        public static int outputs = 10;

        public static int[] hiddenLayersSizes = new int[]
        {
            80,
        };

        public static int getLayerSize(int index)
        {
            if (index == 0)
                return inputs;
            else if (index > hiddenLayersSizes.Length)
                return outputs;
            else
                return hiddenLayersSizes[index - 1];
        }

        public static double random()
        {
            if (R.Next(0, 2) > 0)
                return R.NextDouble();

            return -R.NextDouble();
        }

        public static double activationFunction(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }
    }
}