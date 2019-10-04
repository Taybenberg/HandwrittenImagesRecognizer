using System.IO;
using System.Linq;
using ImageMagick;

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
                images[i] = Directory.GetFiles(path + @"\" + i).ToArray();
                imagesIndex[i] = 0;
            }
        }

        public string GetNextImage(int number)
        {
            if (images[number].Length <= imagesIndex[number])
                imagesIndex[number] = 0;

            return images[number][imagesIndex[number]++];
        }

        public void ConvertImages()
        {
            foreach (var i in images)
                foreach (var img in i)
                    using (MagickImage image = new MagickImage(img))
                    {
                        image.ColorSpace = ColorSpace.Gray;

                        image.Resize(Settings.imageWidth, Settings.imageHeight);

                        image.Write(img);
                    }
        }
    }
}