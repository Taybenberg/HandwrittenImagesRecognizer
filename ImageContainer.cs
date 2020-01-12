using System.IO;
using System.Linq;

namespace HandwrittenNumbersRecognition
{
    public class ImageContainer
    {
        string[][] images = new string[10][];
        int[] imagesIndex = new int[10];

        public ImageContainer(string path)
        {
            for (int i = 0; i < 10; i++)
            {
                images[i] = Directory.GetFiles($"{path}\\{i}").ToArray();
                imagesIndex[i] = 0;
            }
        }

        public string GetNextImage(int number)
        {
            if (images[number].Length <= imagesIndex[number])
                imagesIndex[number] = 0;

            return images[number][imagesIndex[number]++];
        }
    }
}