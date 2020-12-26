using System;
using Microsoft.ML;

namespace HandwrittenNumbersRecognition.ML_NET
{
    class ImageRecognizer
    {
        PredictionEngine<ImageData, ImagePrediction> predictionEngine;

        public ImageRecognizer(string modelFolder)
        {
            var context = new MLContext();
            var model = context.Model.Load(modelFolder, out var modelInputSchema);

            predictionEngine = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
        }

        public void Recognize(string imagePath)
        {
            var image = new ImageData()
            {
                ImagePath = imagePath
            };

            var prediction = predictionEngine.Predict(image);

            Console.WriteLine();
            Console.WriteLine($"ImageFile : [{image.ImagePath}], " +
                                $"Scores : [{string.Join(",", prediction.Score)}], " +
                                $"Predicted Label : {prediction.PredictedLabel}");
        }
    }
}
