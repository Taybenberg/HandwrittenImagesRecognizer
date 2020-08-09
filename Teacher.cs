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

        public void Recognize(string imagePath)
        {
            var output = neuralNet.Output(ImageToDoubleArray(imagePath));

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
            Console.ForegroundColor = ConsoleColor.White;

            int epoch = 0;

            bool flag = true;

            while (flag)
            {
                double[] expected = new double[Settings.outputs];

                int number = Settings.R.Next(0, Settings.outputs);

                expected[number] = 1.0;

                string img = imageContainer.GetNextImage(number);

                double[] output = neuralNet.Output(ImageToDoubleArray(img));

                neuralNet.BackPropagation(expected);

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
                        Console.WriteLine($"\nError: {error}\nOutput number: {max}; Expected number: {number}");

                    if (epoch % 1000 == 0)
                        SaveNet(neuralNet);
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

        double[] ImageToDoubleArray(string path)
        {
            using (var bitmap = new Bitmap(path))
            {
                BitmapData bmpdata = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);

                int numbytes = bmpdata.Stride * bitmap.Height;

                byte[] bytedata = new byte[numbytes];

                Marshal.Copy(bmpdata.Scan0, bytedata, 0, numbytes);

                bitmap.UnlockBits(bmpdata);

                double[] data = new double[bytedata.Length];

                for (int i = 0; i < bytedata.Length; i++)
                    data[i] = bytedata[i] / 255d;

                return data;
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