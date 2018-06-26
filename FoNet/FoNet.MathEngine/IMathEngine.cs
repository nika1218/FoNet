using System.Collections.Generic;
using FoNet.MathEngine.Primitives;

namespace FoNet.MathEngine
{
    public interface IMathEngine
    {
        float[] Multiply(float[] vector, IDictionary<ushort, float[,]> matrices,
            Function activationFunction = Function.Linear);
        float[] Multiply(float[] vector, float[,] matrix,
            Function activationFunction = Function.Linear);

        float[] ApplyFunction(float[] vector, Function activationFunction);
        float[] ApplyDerivativeFunction(float[] vector, Function activationFunction);
        float[] ApplyIntegralFunction(float[] vector, Function activationFunction);
    }
}
