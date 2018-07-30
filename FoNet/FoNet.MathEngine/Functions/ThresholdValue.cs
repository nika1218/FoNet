namespace FoNet.MathEngine.Functions
{
    public class ThresholdValue : IFunction
    {
        private readonly float _zero;
        private readonly float _treshold;
        private readonly float _one;

        public ThresholdValue(float zero, float treshold, float one)
        {
            _zero = zero;
            _treshold = treshold;
            _one = one;
        }

        public float Apply(float val)
        {
            return val < _treshold ? _zero : _one;
        }

        public float Derevative(float val)
        {
            return Apply(val);
        }

        public float Integral(float val)
        {
            return Apply(val);
        }
    }
}
