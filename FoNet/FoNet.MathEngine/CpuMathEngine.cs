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
                    result[y] += ApplyFunction(vector[x] * matrix[x, y], activationFunction);
                }
            });

            return result;
        }

        public float[] Errors(float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length) throw new ArgumentOutOfRangeException(nameof(vector1), $"Vector 1 length {vector1.Length} doesn't correspond to vector 2 length {vector2.Length}");

            var result = new float[vector1.Length];

            Parallel.For(0, vector1.Length, i =>
            {
                result[i] += (float)Math.Pow(vector1[i] - vector2[i], 2);
            });

            return result;
        }

        #region ApplyFunction

        public float[] ApplyFunction(float[] vector, Function activationFunction)
        {
            var result = new float[vector.Length];
            Parallel.For(0, vector.Length, i => { result[i] = ApplyFunction(vector[i], activationFunction); });
            return vector;
        }

        public float[] ApplyDerivativeFunction(float[] vector, Function activationFunction)
        {
            var result = new float[vector.Length];
            Parallel.For(0, vector.Length,
                i => { result[i] = ApplyDerivativeFunction(vector[i], activationFunction); });
            return vector;
        }

        public float[] ApplyIntegralFunction(float[] vector, Function activationFunction)
        {
            var result = new float[vector.Length];
            Parallel.For(0, vector.Length, i => { result[i] = ApplyIntegralFunction(vector[i], activationFunction); });
            return vector;
        }

        public float ApplyFunction(float val, Function activationFunction)
        {
            if (activationFunction == Function.Sigmoid)
            {
                return 1 / (1 + (float)Math.Exp(val));
            }

            return val;
        }

        public float ApplyDerivativeFunction(float val, Function activationFunction)
        {
            if (activationFunction == Function.Sigmoid)
            {
                var sigm = ApplyFunction(val, Function.Sigmoid);
                return sigm * (1 - sigm);
            }

            return val;
        }

        public float ApplyIntegralFunction(float val, Function activationFunction)
        {
            if (activationFunction == Function.Sigmoid)
            {
                return (float)Math.Log(1 + Math.Exp(val));
            }

            return val;
        }

        #endregion

        public float[] CorrectWeights(float[] vector, IDictionary<ushort, float[,]> matrices, float[] ideal,
            Function activationFunction = Function.Linear)
        {
            float[][] outputs = new float[matrices.Count][];

            // forward pass
            var currentVector = vector;
            for (ushort i = 0; i < matrices.Count; i++)
            {
                outputs[i] = Multiply(currentVector, matrices[i], activationFunction);
                currentVector = outputs[i];
            }

            var sigmaOut = SigmaOut(ideal, outputs[matrices.Count - 1], outputs[matrices.Count], activationFunction);

            for (ushort i = (ushort)matrices.Count; i > 0; i--)
            {
                var isLastLayer = i == matrices.Count;

            }

            return vector;
        }

        private float[] SigmaOut(float[] ideal, float[] input, float[] output, Function activationFunction)
        {
            float[] sigmaOut = new float[output.Length];

            var inputDerivative = ApplyDerivativeFunction(input, activationFunction);

            Parallel.For(0, sigmaOut.Length, i => { sigmaOut[i] = (ideal[i] - output[i]) * inputDerivative[i]; });

            return sigmaOut;
        }

        private float[] SigmaHidden(float[] ideal, float[] input, float[] output, Function activationFunction)
        {
            float[] sigmaOut = new float[output.Length];

            Parallel.For(0, sigmaOut.Length, i =>
            {
                sigmaOut[i] = (ideal[i] - output[i]) * ApplyDerivativeFunction(input[i], activationFunction);
            });

            return sigmaOut;
        }
    }
}
