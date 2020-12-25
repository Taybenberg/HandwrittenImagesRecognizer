namespace HandwrittenNumbersRecognition
{
    class Program
    {
        static void Main(string[] args)
        {
            var teacher = new MyNeuralNet.Teacher(@"C:\Users\taybe\source\repos\numbers");
            teacher.Train(true);

            var recognizer = new MyNeuralNet.ImageRecognizer("./MyNeuralNet/NeuralNet.json");
            recognizer.Recognize(@"C:\Users\taybe\source\repos\numbers\0\number-0000000.PNG");
        }
    }
}