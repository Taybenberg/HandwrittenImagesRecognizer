using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Text;
using System.Runtime.InteropServices;
using Newtonsoft.Json;

namespace HandwrittenNumbersRecognition
{
    public class Teacher
    {
        bool printMessages;

        ImageContainer imageContainer;

        NeuralNet neuralNet;

        string netJson = "NeuralNet.json";

        public Teacher()
        {
            neuralNet = LoadNet();
        }

        public void Recoznize(string imagePath)
        {
            var output = neuralNet.Output(ImageToByteArray(imagePath));

            Console.WriteLine($"Image: {imagePath}");

            Console.WriteLine("NeuralNet output: ");

            int max = 0;

            for (int i = 0; i < output.Length; ++i)
            {
                Console.Write($"{output[i]} ");

                if (output[i] > output[max])
                    max = i;
            }

            Console.WriteLine($"\nRecognized number: {max}");
        }

        public Teacher(string datasetPath, bool printMessages = true)
        {
            this.printMessages = printMessages;

            imageContainer = new ImageContainer(datasetPath);

            neuralNet = LoadNet();

            Train();
        }

        void Train()
        {
            int epoch = 0;

            while (true)
            {
                while (!Console.KeyAvailable)
                {
                    int number = Settings.R.Next(0, Settings.outputs);

                    double[] expected = new double[Settings.outputs];
                    expected[number] = 1.0;

                    string img = imageContainer.GetNextImage(number);

                    double[] output = neuralNet.Output(ImageToByteArray(img));

                    neuralNet.BackPropagation(expected);

                    if (epoch % 1000 == 0)
                    {
                        SaveNet(neuralNet);

                        Console.Write($"\n\nEpoch: {epoch}; Image: {img}\nNeuralNet output: ");

                        double error = 0.0;
                        int max = 0;
                        for (int i = 0; i < Settings.outputs; ++i)
                        {
                            error += (output[i] - expected[i]) * (output[i] - expected[i]);
                            if (output[i] > output[max])
                                max = i;
                            Console.Write($"{output[i]} ");
                        }

                        Console.WriteLine($"\nError: {error / 2}\nOutput number: {max}; Expected number: {number}");
                    }
                    else if (printMessages)
                    {
                        Console.Write($"\n\nEpoch: {epoch}; Image: {img}\nNeuralNet output: ");

                        double error = 0.0;
                        int max = 0;
                        for (int i = 0; i < Settings.outputs; ++i)
                        {
                            error += (output[i] - expected[i]) * (output[i] - expected[i]);
                            if (output[i] > output[max])
                                max = i;
                            Console.Write($"{output[i]} ");
                        }

                        Console.WriteLine($"\nError: {error / 2}\nOutput number: {max}; Expected number: {number}");
                    }

                    epoch++;
                }

                var key = Console.ReadKey(true).Key;

                if (key == ConsoleKey.Escape)
                    break;
                else if (key == ConsoleKey.DownArrow)
                    Settings.learningSpeed *= 0.9;
                else if (key == ConsoleKey.UpArrow)
                    Settings.learningSpeed /= 0.9;

                Console.WriteLine($"\nLearning speed: {Settings.learningSpeed}\n");
            }
        }

        byte[] ImageToByteArray(string path)
        {
            using (var bitmap = new Bitmap(path))
            {
                BitmapData bmpdata = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);

                int numbytes = bmpdata.Stride * bitmap.Height;

                byte[] bytedata = new byte[numbytes];

                Marshal.Copy(bmpdata.Scan0, bytedata, 0, numbytes);

                bitmap.UnlockBits(bmpdata);

                return bytedata;
            }
        }

        NeuralNet LoadNet()
        {
            if (File.Exists(netJson))
                return JsonConvert.DeserializeObject<NeuralNet>(File.ReadAllText(netJson, Encoding.UTF8));
                
            return new NeuralNet();
        }

        void SaveNet(NeuralNet neuralNet)
        {
            File.WriteAllText(netJson, JsonConvert.SerializeObject(neuralNet), Encoding.UTF8);
        }
    }
}