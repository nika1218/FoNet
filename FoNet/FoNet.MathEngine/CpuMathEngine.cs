using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using FoNet.MathEngine.Primitives;

namespace FoNet.MathEngine
{
    public class CpuMathEngine : IMathEngine
    {
        public float[] Multiply(float[] vector, IDictionary<ushort, float[,]> matrices, Function activationFunction = Function.Linear)
        {
            var currentVector = vector;

            foreach (var currentMatrix in matrices)
            {
                currentVector = Multiply(currentVector, currentMatrix.Value, activationFunction);
            }

            return currentVector;
        }

        public float[] Multiply(float[] vector, float[,] matrix, Function activationFunction = Function.Linear)
        {
            var xLength = matrix.GetLength(0);
            var yLength = matrix.GetLength(1);

            if (vector.Length != xLength) throw new ArgumentOutOfRangeException(nameof(vector), $"Vector length {vector.Length} doesn't correspond to matrix size [{xLength},{yLength}]");

            var result = new float[yLength];

            Parallel.For(0, yLength, y =>
            {
                for (int x = 0; x < xLength; x++)
                {
                    result[y] += vector[x] * matrix[x, y];
                }
            });

            if (activationFunction == Function.Sigmoid)
            {
                result = Sigmoid(result);
            }

            return result;
        }

        public float[] ApplyFunction(float[] vector, Function activationFunction)
        {
            if (activationFunction == Function.Sigmoid)
            {
                return Sigmoid(vector);
            }

            return vector;
        }

        public float[] ApplyDerivativeFunction(float[] vector, Function activationFunction)
        {
            if (activationFunction == Function.Sigmoid)
            {
                return SigmoidDerivative(vector);
            }

            return vector;
        }

        public float[] ApplyIntegralFunction(float[] vector, Function activationFunction)
        {
            if (activationFunction == Function.Sigmoid)
            {
                return SigmoidIntegral(vector);
            }

            return vector;
        }

        public float[] CorrectWeights(float[] vector, IDictionary<ushort, float[,]> matrices, float[] ideal,
            Function activationFunction = Function.Linear)
        {

            return vector;
        }

        #region Sigmoid

        public float[] Sigmoid(float[] vector)
        {
            var result = new float[vector.Length];

            Parallel.For(0, vector.Length, i => { result[i] = 1 / (1 + (float)Math.Exp(vector[i])); });

            return result;
        }

        public float[] SigmoidDerivative(float[] vector)
        {
            var result = new float[vector.Length];

            Parallel.For(0, vector.Length, i =>
            {
                var sigm = 1 / (1 + (float)Math.Exp(vector[i]));
                result[i] = sigm * (1 - sigm);
            });

            return result;
        }

        public float[] SigmoidIntegral(float[] vector)
        {
            var result = new float[vector.Length];

            Parallel.For(0, vector.Length, i =>
            {
                result[i] = (float)Math.Log(1 + Math.Exp(vector[i]));
            });

            return result;
        }

        #endregion
    }
}
