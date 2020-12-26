namespace HandwrittenNumbersRecognition
{
    class Program
    {
        static void Main(string[] args)
        {
            var teacher = new ML_NET.ImageRecognizerTrainer(@"datasetFolder", @"modelFolder\ML_NET\Model.zip");


            var recognizer = new ML_NET.ImageRecognizer(@"modelFolder\ML_NET\Model.zip");
            recognizer.Recognize(@"imagePath");

            /*
            var teacher = new MyNeuralNet.Teacher(@"datasetFolder", @"modelFolder\MyNeuralNet\NeuralNet.json");
            teacher.Train(true);

            var recognizer = new MyNeuralNet.ImageRecognizer(@"modelFolder\MyNeuralNet\NeuralNet.json");
            recognizer.Recognize(@"imageFolderPath");
            */
        }
    }
}