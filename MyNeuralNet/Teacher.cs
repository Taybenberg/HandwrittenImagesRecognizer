using System;

namespace HandwrittenNumbersRecognition.MyNeuralNet
{
    public class Teacher
    {
        ImageContainer imageContainer;
        ImageRecognizer imageRecognizer;

        public Teacher(string datasetPath, string modelPath)
        {
            imageContainer = new(datasetPath);
            imageRecognizer = new(modelPath);
        }

        public void Train(bool printMessages = false)
        {
            Console.ForegroundColor = ConsoleColor.White;

            int epoch = 0;

            bool flag = true;

            while (flag)
            {
                double[] expected = new double[Settings.outputs];

                int number = Settings.R.Next(0, Settings.outputs);

                expected[number] = 1.0;

                string img = imageContainer.GetNextImage(number);

                double[] output = imageRecognizer.GetNetOutput(img);

                imageRecognizer.NetBackPropagation(expected);

                if (epoch % 1000 == 0)
                    imageRecognizer.SaveNet();

                if (printMessages || epoch % 1000 == 0)
                {
                    Console.Write($"\n\nEpoch: {epoch}; Image: {img}\nNeuralNet output: ");

                    double error = 0.0;

                    int max = 0;

                    for (int i = 0; i < Settings.outputs; i++)
                    {
                        error += ((expected[i] - output[i]) * (expected[i] - output[i])) / 2;

                        if (output[i] > output[max])
                            max = i;

                        Console.Write($"{output[i]} ");
                    }

                    if (expected[max] == 1)
                    {
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.WriteLine($"\nError: {error}\nOutput number: {max}; Expected number: {number}");
                        Console.ForegroundColor = ConsoleColor.White;
                    }
                    else
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine($"\nError: {error}\nOutput number: {max}; Expected number: {number}");
                        Console.ForegroundColor = ConsoleColor.White;
                    }
                }
        
                if (Console.KeyAvailable)
                {
                    switch (Console.ReadKey(true).Key)
                    {
                        case ConsoleKey.Escape:
                            flag = false;
                            break;

                        case ConsoleKey.DownArrow:
                            Settings.learningSpeed *= 0.9;
                            break;

                        case ConsoleKey.UpArrow:
                            Settings.learningSpeed /= 0.9;
                            break;

                        case ConsoleKey.Enter:
                            printMessages = !printMessages;
                            break;
                    }

                    Console.WriteLine($"\nLearning speed: {Settings.learningSpeed}\n");
                }

                epoch++;
            }
        }
    }
}