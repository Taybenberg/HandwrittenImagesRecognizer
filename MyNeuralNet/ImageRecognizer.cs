using System;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace HandwrittenNumbersRecognition.MyNeuralNet
{
    class ImageRecognizer
    {
        NeuralNet neuralNet;
        string modelPath;

        public ImageRecognizer(string modelPath = "NeuralNet.json")
        {
            this.modelPath = modelPath;

            LoadNet();
        }

        public int Recognize(string imagePath)
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

            return max;
        }

        public double[] GetNetOutput(string imagePath)
        {
            return neuralNet.Output(ImageToDoubleArray(imagePath));
        }

        public void NetBackPropagation(double[] expectedOutput)
        {
            neuralNet.BackPropagation(expectedOutput);
        }

        public void LoadNet()
        {
            if (File.Exists(modelPath))
                neuralNet = JsonSerializer.Deserialize<NeuralNet>(File.ReadAllText(modelPath, Encoding.UTF8));
            else
                neuralNet = new();
        }
        public void SaveNet()
        {
            File.WriteAllText(modelPath, JsonSerializer.Serialize(neuralNet), Encoding.UTF8);
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
    }
}