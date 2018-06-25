namespace FoNet.ANN
{
    public class Layer
    {
        public ushort NeuronsCount { get; }

        public Layer(ushort neuronsCount)
        {
            NeuronsCount = neuronsCount;
        }
    }
}
