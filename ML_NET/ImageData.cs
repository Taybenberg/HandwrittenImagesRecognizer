using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace HandwrittenNumbersRecognition.ML_NET
{
    class ImageData
    {
        public string ImagePath;
        public uint Label;

        public static IEnumerable<ImageData> GetImageData(string path)
        {
            List<IEnumerable<ImageData>> data = new();

            Parallel.For(0, 10, (i) =>
            {
                data.Add(Directory.GetFiles($"{path}\\{i}").
                    Select(file => new ImageData()
                    {
                        ImagePath = file,
                        Label = Convert.ToUInt32(i)
                    }));
            });

            return data.SelectMany(x => x);
        }
    }
}
