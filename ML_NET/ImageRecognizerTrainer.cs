using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

namespace HandwrittenNumbersRecognition.ML_NET
{
    class ImageRecognizerTrainer
    {
        public ImageRecognizerTrainer(string imagesFolder, string modelFolder)
        {
            Console.WriteLine("Loading images...");

            var images = ImageData.GetImageData(imagesFolder);

            Console.WriteLine($"Done. Loaded {images.Count()} images.");
            Console.WriteLine("MLContext initialization...");

            MLContext context = new();
            //var model = context.Model.Load(modelFolder, out var inputSchema);

            var fullImagesDataset = context.Data.LoadFromEnumerable(images);
            var shuffledImagesDataset = context.Data.ShuffleRows(fullImagesDataset);

            Console.WriteLine("Done.");
            Console.WriteLine("Pipeline definition...");

            var preProcessPipeline = context.Transforms.Conversion.MapValueToKey("LabelAsKey", "Label")
                .Append(context.Transforms.LoadRawImageBytes("Image", imagesFolder, "ImagePath"));

            var preProcessData = preProcessPipeline.Fit(shuffledImagesDataset).Transform(shuffledImagesDataset);

            var trainTestData = context.Data.TrainTestSplit(preProcessData, 0.1);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = testData,
                Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true
            };

            var pipeline = context.MulticlassClassification.Trainers
                .ImageClassification(classifierOptions)
                .Append(context.Transforms.Conversion.MapValueToKey("PredictedLabel", "PredictedLabel"));

            Console.WriteLine("Done.");
            Console.WriteLine("Starting training...");

            var trainedModel = pipeline.Fit(trainData);

            context.Model.Save(trainedModel, trainData.Schema, modelFolder);

            Console.WriteLine("Done. Model Saved.");
            Console.WriteLine("Calculating model metrics...");

            var predictionsData = trainedModel.Transform(testData);
            var metrics = context.MulticlassClassification.Evaluate(predictionsData, "LabelAsKey", predictedLabelColumnName: "PredictedLabel");

            PrintMultiClassClassificationMetrics("TensorFlow DNN Transfer Learning", metrics);

            Console.ReadLine();
        }

        void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");

            int i = 0;
            foreach (var classLogLoss in metrics.PerClassLogLoss)
                Console.WriteLine($"    LogLoss for class {i++} = {classLogLoss:0.####}, the closer to 0, the better");

            Console.WriteLine($"************************************************************");
        }
    }
}