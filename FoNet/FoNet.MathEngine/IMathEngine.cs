using System.Collections.Generic;
using FoNet.MathEngine.Functions;
using FoNet.MathEngine.Primitives;

namespace FoNet.MathEngine
{
    public interface IMathEngine
    {
        float[] Multiply(float[] vector, IDictionary<ushort, float[,]> matrices,
            IFunction activationFunction);
        float[] Multiply(float[] vector, float[,] matrix,
            IFunction activationFunction);

        float[] ApplyFunction(float[] vector, IFunction activationFunction);
        float[] ApplyDerivativeFunction(float[] vector, IFunction activationFunction);
        float[] ApplyIntegralFunction(float[] vector, IFunction activationFunction);

        float[] CorrectWeightsIteration(float epsilon, float[] vector, IDictionary<ushort, float[,]> matrices, float[] ideal,
            IFunction activationFunction);
    }
}
