using System;

namespace FoNet.MathEngine.Functions
{
    public class Sigmoid : IFunction
    {
        public float Apply(float val)
        {
            return 1 / (1 + (float)Math.Exp(val));
        }

        public float Derevative(float val)
        {
            var sigm = Apply(val);
            return sigm * (1 - sigm);
        }

        public float Integral(float val)
        {
            return (float)Math.Log(1 + Math.Exp(val));
        }
    }
}
