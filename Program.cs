namespace HandwrittenNumbersRecognition
{
    class Program
    {
        static void Main(string[] args)
        {
            new Teacher(yourDatasetPath);
            new Teacher().Recognize(yourImagePath);
        }
    }
}